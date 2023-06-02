""" Adapted from https://github.com/Khrylx/AgentFormer/tree/main """
import torch
import numpy as np
from einops import rearrange

from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from .agentformer_loss import loss_func
from .common.dist import *
from .agentformer_lib import AgentFormerEncoderLayer, AgentFormerDecoderLayer, AgentFormerDecoder, AgentFormerEncoder
from .map_encoder import MapEncoder


class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        initialize_weights(self.affine_layers.modules())        

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)


def rotation_2d_torch(x, theta, origin=None):
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x


def generate_ar_mask(sz, num_agent, agent_mask):
    assert sz % num_agent == 0
    T = sz // num_agent
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * num_agent
        i2 = (t+1) * num_agent
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_mask(tgt_sz, src_sz, num_agent, agent_mask):
    assert tgt_sz % num_agent == 0 and src_sz % num_agent == 0
    mask = agent_mask.repeat(tgt_sz // num_agent, src_sz // num_agent)
    return mask



class ExpParamAnnealer(nn.Module):
    def __init__(self, start, finish, rate, cur_epoch=0):
        super().__init__()
        self.register_buffer('start', torch.tensor(start))
        self.register_buffer('finish', torch.tensor(finish))
        self.register_buffer('rate', torch.tensor(rate))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.finish - (self.finish - self.start) * (self.rate ** self.cur_epoch)


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):    
    def __init__(self, cfg):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=cfg["tf_dropout"])
        self.concat = cfg["pos_concat"]
        self.d_model = cfg["tf_model_dim"]
        self.use_agent_enc = cfg["use_agent_enc"]
        self.agent_enc_learn = cfg["agent_enc_learn"]
        if self.concat:
            # concat time_emd and/or agent_emd
            self.fc = nn.Linear((3 if self.use_agent_enc else 2) * self.d_model, self.d_model)

        pe = self.build_pos_enc(cfg["max_t_len"])
        self.register_buffer('pe', pe)
        if self.use_agent_enc:
            if self.agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(cfg["max_a_len"], 1, self.d_model) * 0.1)
            else:
                ae = self.build_pos_enc(cfg["max_a_len"])
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        # pos_enc (num_t * num_a, 1, d_model)
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            # agent_enc (num_t * num_a, 1, d_model)
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


""" Context (Past) Encoder """
class ContextEncoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model_dim = cfg['tf_model_dim']
        self.ff_dim = cfg['tf_ff_dim']
        self.nhead = cfg['tf_nhead']
        self.dropout = cfg['tf_dropout']
        self.nlayer = cfg['context_encoder']['nlayer']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = cfg['agent_enc_shuffle'] if cfg['use_agent_enc'] else None
        cfg['context_dim'] = self.model_dim
        in_dim = cfg['obs_size']

        self.input_fc = nn.Linear(in_dim, self.model_dim)
        self.pos_encoder = PositionalAgentEncoding(cfg)
        encoder_layers = AgentFormerEncoderLayer(cfg['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)

    def forward(self, data, agent_mask):        
        num_agent = self.cfg['num_agent']
        tf_in = self.input_fc(data)     
        tf_in_pos = self.pos_encoder(tf_in, num_a=num_agent, agent_enc_shuffle=self.agent_enc_shuffle)
                
        src_agent_mask = agent_mask.clone()        
        src_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], num_agent, src_agent_mask).to(tf_in.device)
        # aggregate context across all agents
        context_enc = self.tf_encoder(tf_in_pos, mask=src_mask, num_agent=num_agent)        
        context_rs = context_enc.reshape(-1, num_agent, self.model_dim)
        # compute per agent context
        if self.pooling == 'mean':
            agent_context = torch.mean(context_rs, dim=0)
        else:
            agent_context = torch.max(context_rs, dim=0)[0]
        return context_enc, agent_context


""" Future Encoder """
class FutureEncoder(nn.Module):
    def __init__(self, cfg, **kwargs): 
        super().__init__()
        self.cfg = cfg
        # this is the same as model dimension set in ContextEncoder
        self.context_dim = cfg['context_dim']
        self.forecast_dim = cfg['obs_size']
        self.nz = cfg['nz']
        self.z_type = cfg['z_type']
        if 'z_tau_annealer' in cfg:
            self.z_tau_annealer = cfg['z_tau_annealer']
        else:
            self.z_tau_annealer = None
        self.model_dim = cfg['tf_model_dim']
        self.ff_dim = cfg['tf_ff_dim']
        self.nhead = cfg['tf_nhead']
        self.dropout = cfg['tf_dropout']
        self.nlayer = cfg['future_encoder']['nlayer']
        self.out_mlp_dim = cfg['future_decoder']['out_mlp_dim']
        self.pooling = cfg['pooling']
        self.agent_enc_shuffle = cfg['agent_enc_shuffle']
        # networks
        in_dim = self.cfg['obs_size']        
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(cfg['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(cfg)
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, traj_in, context_enc, agent_mask, reparam=True):        
        num_agent = self.cfg['num_agent']    
        # (T * A, B, E)
        fc_traj_in = rearrange(traj_in, 't a b d -> (t a) b d')
        tf_in = self.input_fc(fc_traj_in)
        agent_enc_shuffle = self.cfg['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        # (T*A, B, E)
        tf_in_pos = self.pos_encoder(tf_in, num_a=num_agent, agent_enc_shuffle=agent_enc_shuffle)        

        # (A, A)
        mem_agent_mask = agent_mask.clone()
        # (A, A)
        tgt_agent_mask = agent_mask.clone()
        # (T*A, past_T*A)
        mem_mask = generate_mask(tf_in.shape[0], context_enc.shape[0], num_agent, mem_agent_mask).to(tf_in.device)
        # (T*A, T*A)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], num_agent, tgt_agent_mask).to(tf_in.device)        
        
        # (T*A, B, E)        
        tf_out, _ = self.tf_decoder(tf_in_pos, context_enc, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=self.cfg['num_agent'])
        #tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)
        tf_out = tf_out.reshape(-1, traj_in.shape[1], self.model_dim)        
        
        if self.pooling == 'mean':
            # (A, E)
            h = torch.mean(tf_out, dim=0)
        else:
            h = torch.max(tf_out, dim=0)[0]
        if self.out_mlp_dim is not None:
            # (A, E)
            h = self.out_mlp(h)

        # (A, num_dist_params)        
        q_z_params = self.q_z_net(h)        
        if self.z_type == 'gaussian':
            q_z_dist = Normal(params=q_z_params)
        else:
            q_z_dist = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        q_z_samp = q_z_dist.rsample()
        return q_z_dist, q_z_samp


""" Future Decoder """
class FutureDecoder(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = cfg['ar_detach']
        self.context_dim = cfg['context_dim']
        self.forecast_dim = forecast_dim = cfg['obs_size']
        self.pred_scale = cfg['pred_scale']
        self.future_frames = cfg['future_frames']
        self.past_frames = cfg['past_frames']
        self.nz = cfg['nz']
        self.z_type = cfg['z_type']
        self.model_dim = cfg['tf_model_dim']
        self.ff_dim = cfg['tf_ff_dim']
        self.nhead = cfg['tf_nhead']
        self.dropout = cfg['tf_dropout']
        self.nlayer = cfg['future_decoder']['nlayer']
        self.out_mlp_dim = cfg['future_decoder']['out_mlp_dim']
        self.pos_offset = cfg['pos_offset']
        self.agent_enc_shuffle = cfg['agent_enc_shuffle']
        self.learn_prior = cfg['learn_prior']
        # networks
        in_dim = self.forecast_dim + self.nz        
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer(cfg['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(cfg)
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, forecast_dim)
        else:
            in_dim = self.model_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())
    
    def decode_traj_ar(self, dec_in, mode, context, agent_mask, z, sample_num, need_weights=False):
        num_agent = self.cfg['num_agent']
        # orginal shape (T, A*num_sample, B, F)        
        bs = dec_in.shape[-2]        
        dec_in = rearrange(dec_in[[-1]], 't a b f -> a t b f')
        # (A, T, B, F) -> (A, num_sample*B, F)
        dec_in = dec_in.reshape(dec_in.shape[0], -1, dec_in.shape[-1])

        # z: (A * num_sample, dz) -> (A, num_sample * B, dz) 
        z_in = z.reshape(-1, sample_num, z.shape[-1]).repeat_interleave(bs, dim=1)
        in_arr = [dec_in, z_in]
        # (A, num_sample * B, F+dz)       
        dec_in_z = torch.cat(in_arr, dim=-1)        

        mem_agent_mask = agent_mask.clone()
        tgt_agent_mask = agent_mask.clone()

        for i in range(self.future_frames):
                
            # (B, A, num_sample, E)            
            tf_in = self.input_fc(dec_in_z)            
            agent_enc_shuffle = self.cfg['agent_enc_shuffle'] if self.agent_enc_shuffle else None

            tf_in_pos = self.pos_encoder(tf_in, num_a=num_agent, agent_enc_shuffle=agent_enc_shuffle, t_offset=self.past_frames-1 if self.pos_offset else 0)            
            # tf_in_pos = tf_in
            mem_mask = generate_mask(tf_in.shape[0], context.shape[0], self.cfg['num_agent'], mem_agent_mask).to(tf_in.device)
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], num_agent, tgt_agent_mask).to(tf_in.device)

            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=self.cfg['num_agent'], need_weights=need_weights)            
            out_tmp = tf_out.view(-1, tf_out.shape[-1])
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(out_tmp)
            seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
            if self.ar_detach:
                out_in = seq_out[-num_agent:].clone().detach()
            else:
                out_in = seq_out[-num_agent:]
            
            # create dec_in_z
            in_arr = [out_in, z_in]            
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, num_agent * sample_num, seq_out.shape[-1])
        return seq_out, attn_weights
        

    def decode_traj_batch(self, data, mode, context, agent_mask, z, sample_num):
        raise NotImplementedError

    def forward(self, dec_in, mode, context_enc, agent_context, agent_mask, q_z_dist, q_z_samp, sample_num=1, autoregress=True, need_weights=False):        
        num_agent = self.cfg['num_agent']
        # Pre dec_in: (T*A, B, E) -> (T, A, B, E)
        dec_in = dec_in.reshape(-1, num_agent, dec_in.shape[1], dec_in.shape[-1])        
        # (T, A * sample_num, B, E)
        dec_in = dec_in.repeat_interleave(sample_num, dim=1)
        # (A * T, B * sample_num, E)
        context = context_enc.repeat_interleave(sample_num, dim=1)        

        # p(z)        
        if self.learn_prior:
            # agent_context: (A, E), h: (A * num_sample, E)
            h = agent_context.repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                p_z_dist = Normal(params=p_z_params)
            else:
                p_z_dist = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                p_z_dist = Normal(mu=torch.zeros(dec_in.shape[1], self.nz).to(dec_in.device), logvar=torch.zeros(dec_in.shape[1], self.nz).to(dec_in.device))
            else:
                p_z_dist = Categorical(logits=torch.zeros(dec_in.shape[1], self.nz).to(dec_in.device))

        if mode in {'train', 'recon'}:
            q_z_dist.mode()
            z = q_z_samp if mode == 'train' else q_z_dist.mode()
        elif mode == 'infer':
            z = p_z_dist.sample()
        else:
            raise ValueError('Unknown Mode!')
                
        if autoregress:           
            seq_out, _ = self.decode_traj_ar(dec_in, mode, context, agent_mask, z, sample_num, need_weights=need_weights)
            return seq_out, p_z_dist
        else:
            self.decode_traj_batch(dec_in, mode, context, agent_mask, z, sample_num)
        


""" AgentFormer """
class AgentFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        
        self.rand_rot_scene = cfg['rand_rot_scene']
        self.discrete_rot = cfg['discrete_rot']
        self.map_global_rot = cfg['map_global_rot']
        self.ar_train = cfg['ar_train']
        self.max_train_agent = cfg['max_train_agent']
        self.loss_cfg = self.cfg["loss_cfg"]
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        if self.cfg['z_type'] == 'discrete':
            self.cfg['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        # save all computed variables
        self.data = None        

        # models
        self.context_encoder = ContextEncoder(cfg)
        self.future_encoder = FutureEncoder(cfg)
        self.future_decoder = FutureDecoder(cfg)
        
    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data):
        device = self.device
        # rotate the scene
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2
        else:
            theta = torch.zeros(1).to(device)        

        # agent shuffling
        if self.training and self.cfg['agent_enc_shuffle']:
            self.cfg['agent_enc_shuffle'] = torch.randperm(self.cfg['max_a_len'])[:self.cfg['num_agent']].to(device)
        else:
            self.cfg['agent_enc_shuffle'] = None   

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self, data):
        num_agent = self.cfg['num_agent']  
        past_trajs = data[:, :self.cfg['past_frames']].reshape(
            data.size(0), self.cfg['past_frames'], num_agent, -1)
        fut_trajs = data[:, -self.cfg['future_frames']:].reshape(
            data.size(0), self.cfg['future_frames'], num_agent, -1)
        # reshape data to (time, batch, feature)
        past_trajs = rearrange(past_trajs, 'b t a f  -> (t a) b f')
        fut_trajs = rearrange(fut_trajs, 'b t a f  -> t a b f')
        agent_mask = torch.zeros([num_agent, num_agent]).to(data.device)
        
        context_enc, agent_context = self.context_encoder(past_trajs, agent_mask)       
        q_z_dist, q_z_samp = self.future_encoder(fut_trajs, context_enc, agent_mask)        
        seq_out, p_z_dist = self.future_decoder(past_trajs, 'train', context_enc, agent_context, agent_mask, q_z_dist, q_z_samp, autoregress=self.ar_train)

        #if self.compute_sample:
        #    self.inference(context_enc, sample_num=self.loss_cfg['sample']['k'])
        return seq_out, fut_trajs, q_z_dist, p_z_dist

    def inference(self, past_traj, context_enc=None, mode='infer', sample_num=20, need_weights=False):
        if context_enc is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(past_traj)
        self.future_decoder(past_traj, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

    def compute_mse(self, pred_traj, gt_traj):
        pred_traj = pred_traj.reshape(-1, pred_traj.size(-1))
        gt_traj = gt_traj.reshape(-1, gt_traj.size(-1))
        loss_unweighted = F.mse_loss(pred_traj, gt_traj, reduction='mean')
        loss = loss_unweighted * self.cfg['loss_cfg']['mse']['weight']
        return loss, loss_unweighted
    
    def compute_kl(self, q_z_dist, p_z_dist):
        min_clip = self.cfg['loss_cfg']['kld']['min_clip']
        loss_unweighted = q_z_dist.kl(p_z_dist).mean()    # normalize by batch_size
        loss_unweighted = loss_unweighted.clamp_min_(min_clip)
        loss = loss_unweighted * self.cfg['loss_cfg']['kld']['weight']
        return loss, loss_unweighted
    
    def compute_sample_loss(self, pred_traj, gt_traj):
        diff = pred_traj - gt_traj.unsqueeze(1)
        dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
        loss_unweighted = dist.min(dim=1)[0]        
        # normalize the loss
        loss_unweighted = loss_unweighted.mean()
        loss = loss_unweighted * self.cfg['loss_cfg']['sample']['weight']
        return loss, loss_unweighted

    def loss(self, data):
        seq_out, fut_trajs, q_z_dist, p_z_dist = self(data)
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            if loss_name == 'mse':
                loss, loss_unweighted = self.compute_mse(seq_out, fut_trajs)
            elif loss_name == 'kld':
                loss, loss_unweighted = self.compute_kl(q_z_dist, p_z_dist)
            #elif loss_name == 'sample':
            #    loss, loss_unweighted = self.compute_sample_loss(seq_out, fut_trajs)
            total_loss += loss            
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss#, loss_dict, loss_unweighted_dict
