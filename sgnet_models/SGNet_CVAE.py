import torch
import torch.nn as nn
from einops import rearrange
from .bitrap_np import BiTraPNP
import torch.nn.functional as F


class rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''
    def __init__(self):
        super(rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=2)
        # mean of each frames predictions
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        # sum of all batches
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred


def cvae_multi(pred_traj, target, first_history_index = 0):
    '''
    CVAE loss use best-of-many
    '''
    K = pred_traj.shape[3]
    target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
    total_loss = []
    for enc_step in range(first_history_index, pred_traj.size(1)):
        traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        total_loss.append(loss_traj)
    return sum(total_loss)/len(total_loss)


class SGNet_CVAE(nn.Module):
    def __init__(self, config):
        super(SGNet_CVAE, self).__init__()
        self.cvae = BiTraPNP(config)
        self.input_dim = config["obs_size"]
        self.hidden_size = config["hidden_size"] # GRU hidden size
        # This doesn't have to instatiate unless it's training
        if "enc_steps" in config:
            self.enc_steps = config["enc_steps"] # observation step
        if "dec_steps" in config:            
            self.dec_steps = config["dec_steps"] # prediction step
        self.dropout = config["dropout"]
        self.feature_extractor = nn.Sequential(nn.Linear(self.input_dim, self.hidden_size),
                                              nn.ReLU(inplace=True))
        self.pred_dim = config["pred_dim"]        
        self.K = config["K"] # number of trajectory        
        # loss
        self.goal_loss = rmse_loss()
        # the predict shift is in meter
        self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4, 1),
                                           nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4, 1),
                                           nn.ReLU(inplace=True))
        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//4), 
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4, self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + config["latent_dim"], self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), 
                                               nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4, self.hidden_size//4), 
                                                  nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                 nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4, self.hidden_size//4),
                                         nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4, self.hidden_size//4),
                                         nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
    
    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj

    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(self.enc_steps):            
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            # goal_for_dec: a list of goal hiddens for each decoding time step
            # goal_for_enc: goal hidden for the current encoding time step
            # goal_traj: goal trajectories (full dec_length) for the current encoding time step
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:,enc_step,:,:] = goal_traj            
            # aggregated decode hidden
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)            
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(                    
                    dec_hidden, raw_inputs[:,enc_step,:], self.K, 
                    raw_targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(
                    dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities

    def compute_enc_targets(self, inputs):
        enc_targets = []
        for i in range(self.enc_steps):
            start_idx = i + 1
            end_idx = start_idx + self.dec_steps
            enc_targets.append(inputs[:, start_idx : end_idx, :])
        enc_targets = torch.stack(enc_targets, dim=1)
        return enc_targets

    def forward(self, inputs, targets = None, training=True):
        self.training = training
        traj_input = self.feature_extractor(inputs[:, : self.enc_steps, :])
        # all_goal_traj: goal trajectories (full dec_length) for all encoding 
        # time steps (batch_size, enc_steps, dec_steps, pred_dim)
        # all_cvae_dec_traj: sampled cvae decoded trajectories (full dec_length) 
        # for all encoding time steps (batch_size, enc_steps, dec_steps, K, pred_dim)
        # KLD: total KL divergence
        all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None)        
        return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities
    
    def loss(self, inputs):
        loss_dict = {}
        targets = self.compute_enc_targets(inputs)
        all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self(inputs, targets=targets, training=True)
        cvae_loss = cvae_multi(all_cvae_dec_traj, targets)
        goal_loss = self.goal_loss(all_goal_traj, targets)
        kld_loss = KLD.mean()
        total_loss = goal_loss + cvae_loss  + kld_loss
        # log loss
        loss_dict = {
            'total_loss': total_loss,
            'goal_loss': goal_loss.cpu().detach().numpy(),
            'cvae_loss': cvae_loss.cpu().detach().numpy(),
            'kld_loss': kld_loss.cpu().detach().numpy()
        }
        return total_loss, loss_dict
    
    def forward_rollout(self, x, enc_steps, dec_steps):
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps        
        # this will produce the K trajectories with different probabilities
        all_goal_traj, cvae_dec_traj, KLD_loss, total_probabilities = self(x, training=False)
        # max prob index out of the K trajectories
        max_prob_idx = total_probabilities.mean(dim=1).argmax().item()        
        rollouts = cvae_dec_traj[:, -1, :, max_prob_idx, :]
        return rollouts
