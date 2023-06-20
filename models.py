# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:09:25 2022

@author: locro
"""
from collections import namedtuple
from einops import rearrange
from kv_caching import KeysValues, KVCache
from typing import Tuple, Dict, Optional
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as td
from typing import Optional
    
RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter']) 

class DreamerV2(nn.Module):
    def __init__(self, config): 
        super(DreamerV2, self).__init__()
        self.input_size = config['input_size']
        self.deter_size = config['deter_size']
        self.dec_hidden_size = config['dec_hidden_size']
        self.rssm_type = config['rssm_type']
        self.rnn_type = config['rnn_type']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if 'dec_num_layers' in config:
            self.dec_num_layers = config['dec_num_layers']
        else:
            self.dec_num_layers = 3
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1
        if 'dec_dist' in config:
            self.dec_dist = config['dec_dist']
        else:
            self.dec_dist = None
        if 'kl_balancing' in config:
            self.kl_balancing = config['kl_balancing']
            self.kl_alpha = config['kl_alpha']
        else:
            self.kl_balancing = False
            
        if self.rssm_type == 'discrete':
            self.class_size = config['class_size']
            self.category_size = config['category_size']
            self.stoch_size = self.class_size*self.category_size
            self.fc_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.fc_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
            self.rssm_tuple = RSSMDiscState
        elif self.rssm_type == 'continuous':
            self.stoch_size = config['stoch_size']
            self.mean_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.log_var_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.mean_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
            self.log_var_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
            self.rssm_tuple = RSSMContState
            
        # Encoder Part
        if self.rnn_type == 'GRU':
            self.encoder_rnn = nn.GRU(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
            
        # Decoder Part
        self.decoder_mlp = self.build_mlp(self.dec_num_layers, self.stoch_size+self.deter_size, self.dec_hidden_size, self.input_size, nn.ELU)
        
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)
    
    def init_d_states(self, batch_size):
        if self.rnn_type == 'GRU':
            hidden_cell = torch.zeros(1, batch_size, self.deter_size).to(self.device)
            return hidden_cell
        elif self.rnn_type == 'LSTM':
            hidden_cell = torch.zeros(1, batch_size, self.deter_size).to(self.device)
            state_cell = torch.zeros(1, batch_size, self.deter_size).to(self.device)
            return (hidden_cell, state_cell)
    
    def init_s_states(self, batch_size):
        z = torch.zeros(batch_size, 1, self.stoch_size).to(self.device)
        return z
    
    def encoder(self, x):
        self.batch_size = x.size(0)
        self.num_steps = x.size(1)
        hidden_encoder = self.init_d_states(self.batch_size) 
        z_post_in = self.init_s_states(self.batch_size)
        
        priors = []
        posteriors = []
        for t in range(x.size(1)):
            xt = x[:,t,:]
            xt = xt.reshape(xt.size(0), xt.size(1))
            
            output_encoder, hidden_encoder = self.encoder_rnn(z_post_in, hidden_encoder)
            
            if self.rnn_type == 'GRU':
                deter_state = hidden_encoder.view(-1, self.deter_size)
            elif self.rnn_type == 'LSTM':
                deter_state = hidden_encoder[0].view(-1, self.deter_size)
                
            if self.rssm_type == 'discrete':
                # Estimate prior stochastic state
                prior_logit = self.fc_prior(deter_state)
                stats = {'logit': prior_logit}
                z_prior = self.get_stoch_state(stats)
                prior_rssm_state = RSSMDiscState(prior_logit, z_prior, deter_state)
                priors.append(prior_rssm_state)
                
                # Estimate posterior stochastic state
                post_input = torch.cat((xt, deter_state), dim=1)
                post_logit = self.fc_post(post_input)
                stats = {'logit': post_logit}
                z_post = self.get_stoch_state(stats)
                post_rssm_state = RSSMDiscState(post_logit, z_post, deter_state)
                posteriors.append(post_rssm_state)
            elif self.rssm_type == 'continuous':
                # Estimate prior stochastic state
                prior_input = deter_state
                mean_prior = self.mean_prior(prior_input)
                log_var_prior = self.log_var_prior(prior_input)
                std_prior = torch.exp(0.5 * log_var_prior)
                noise_prior = torch.randn(self.batch_size, self.stoch_size).to(self.device)
                z_prior = noise_prior * std_prior + mean_prior
                prior_rssm_state = RSSMContState(mean_prior, std_prior, z_prior, deter_state)
                priors.append(prior_rssm_state)
                
                # Estimate posterior stochastic state
                post_input = torch.cat((xt, deter_state), dim=1)
                mean_post = self.mean_post(post_input)
                log_var_post = self.log_var_post(post_input)
                std_post = torch.exp(0.5 * log_var_post)
                noise_post = torch.randn(self.batch_size, self.stoch_size).to(self.device)
                z_post = noise_post * std_post + mean_post
                post_rssm_state = RSSMContState(mean_post, std_post, z_post, deter_state)
                posteriors.append(post_rssm_state)

            #print(f"{self.rssm_type} noise_prior mean {noise_prior.mean()} std {noise_prior.std()}")
            #print(f"noise_post mean {noise_post.mean()} std {noise_post.std()}")
            #print(f"z_prior mean {z_prior.mean()} std {z_prior.std()}")

            # use z for input of next rnn iteration, resized for (B, T, Z)            
            z_post_in = z_post.reshape(z_post.size(0), 1, z_post.size(1))
        
        prior = self.rssm_stack_states(priors, dim=1)
        post = self.rssm_stack_states(posteriors, dim=1)
        
        return prior, post
    
    def decoder(self, posterior):
        rssm_state = torch.cat((posterior.deter, posterior.stoch), dim=-1)
        dist_inputs = self.decoder_mlp(rssm_state)
        if self.dec_dist == 'normal':
            return td.independent.Independent(td.Normal(dist_inputs, 1), len(self._output_shape))
        if self.dec_dist == None:
            x_hat = dist_inputs
            return x_hat
        
    def rssm_stack_states(self, rssm_states, dim):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.stack([state.logit for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
            torch.stack([state.mean for state in rssm_states], dim=dim),
            torch.stack([state.std for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.deter for state in rssm_states], dim=dim),
        )
        
    def kl_loss(self, prior, posterior):
        prior_dist = self.get_dist(prior)
        post_dist = self.get_dist(posterior)
        if self.kl_balancing:
            prior_nograd = self.rssm_tuple(*(x.detach() for x in prior))
            post_nograd = self.rssm_tuple(*(x.detach() for x in posterior))
            prior_dist_nograd = self.get_dist(prior_nograd)
            post_dist_nograd = self.get_dist(post_nograd)
            kl_loss = self.kl_alpha * torch.mean(torch.distributions.kl.kl_divergence(post_dist_nograd, prior_dist)) \
                + (1 - self.kl_alpha) * torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist_nograd))
        else:
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        
        return kl_loss
    
    def get_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit.shape
            logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)
        
    def get_stoch_state(self, stats):
        if self.rssm_type == 'discrete':
            logit = stats['logit']
            shape = logit.shape
            logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
            dist = td.OneHotCategorical(logits=logit)        
            stoch = dist.sample()
            stoch += dist.probs - dist.probs.detach()
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)
        
    def get_imagined_states(self, prior, num_steps):
        deter_state = prior.deter[:,-1,:]
        hidden_encoder = deter_state.unsqueeze(0)
        z_prior = prior.stoch[:,-1,:]
        z_prior = z_prior.unsqueeze(1)
        
        priors_imagined = []
        for step in range(num_steps):
            output_encoder, hidden_encoder = self.encoder_rnn(z_prior, hidden_encoder.contiguous())
            if self.rnn_type == 'GRU':
                deter_state = hidden_encoder.view(-1, hidden_encoder.size(2))
            elif self.rnn_type == 'LSTM':
                deter_state = hidden_encoder[0].view(-1, hidden_encoder.size(2))
            if self.rssm_type == 'discrete':
                prior_logit = self.fc_prior(deter_state)
                stats = {'logit': prior_logit}
                z_prior = self.get_stoch_state(stats)
                prior_rssm_state = RSSMDiscState(prior_logit, z_prior, deter_state)
            elif self.rssm_type == 'continuous':
                mean_prior = self.mean_prior(deter_state)
                log_var_prior = self.log_var_prior(deter_state)
                std_prior = torch.exp(0.5 * log_var_prior)
                noise_prior = torch.randn(self.batch_size, self.stoch_size).to(self.device)
                z_prior = noise_prior * std_prior + mean_prior
                prior_rssm_state = RSSMContState(mean_prior, std_prior, z_prior, deter_state)
            priors_imagined.append(prior_rssm_state)
            # use z for input of next rnn iteration, resized for (B, T, Z) 
            z_prior = z_prior.reshape(z_prior.size(0), 1, z_prior.size(1))
    
        priors_imag = self.rssm_stack_states(priors_imagined, dim=1)
        x_hat_imag = self.decoder(priors_imag)
        
        return x_hat_imag
        
    def forward_rollout(self, x, burn_in_length, rollout_length):
        x_burnin = x[:,:burn_in_length,:]
        prior, post = self.encoder(x_burnin)
        x_hat_imag = self.get_imagined_states(prior, rollout_length)
        
        return x_hat_imag
    
    def forward(self, x):
        # Encoder
        prior, post = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder(post)
        
        return x_hat
    
    def loss(self, x):
        # Encoder
        prior, post = self.encoder(x)
        
        # KL loss
        kl_loss = self.kl_loss(prior, post)
        
        # Decoder
        x_hat = self.decoder(post)
        
        self.recon_loss = ((x - x_hat)**2).sum()
        self.kl = self.beta*kl_loss
        elbo = self.recon_loss + self.kl

        return elbo
    
    
class MultistepPredictor(nn.Module):
    def __init__(self, config):
        super(MultistepPredictor, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        if 'input_embed_size' in config:
            self.input_embed_size = config['input_embed_size']
            self.rnn_input_size = self.input_embed_size
        else:
            self.input_embed_size = None
            self.rnn_input_size = self.input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.input_embed_size is not None:
            self.input_embed = nn.Linear(in_features=self.input_size, out_features=self.input_embed_size)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.rnn_input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.input_size, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x, hidden):    
        if self.input_embed_size is not None:
            x = self.input_embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.mlp(out)
        return out, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return hidden_cell
        elif self.rnn_type == 'LSTM':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            state_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return (hidden_cell, state_cell)
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x[:,:burn_in_length,:], hidden)
        # last output is next input x
        xt_hat = output[:,-1,:].unsqueeze(1)
        pred_outputs = []
        for t in range(rollout_length):
            pred_outputs.append(xt_hat.squeeze(1))
            # output is next input x
            xt_hat, hidden = self.forward(xt_hat, hidden)
        if len(pred_outputs) == 0:
            breakpoint()
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def loss(self, x, burn_in_length, rollout_length):
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
    
    
class MultistepDelta(nn.Module):
    def __init__(self, config):
        super(MultistepDelta, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(self.rnn_hidden_size, self.mlp_hidden_size)
        self.fc2 = nn.Linear(self.mlp_hidden_size, self.input_size)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
        elif self.rnn_type == 'LSTM':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            state_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return (hidden_cell, state_cell)
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        x_burnin = x[:,:burn_in_length,:]
        output, hidden = self.forward(x_burnin, hidden)
        # last output is the delta for the next input x
        xt_hat = torch.add(x_burnin[:,-1,:].unsqueeze(1), output[:,-1,:].unsqueeze(1))
        pred_outputs = []
        for t in range(rollout_length):
            pred_outputs.append(xt_hat.squeeze(1))
            # output is next input x
            xt_delta_hat, hidden = self.forward(xt_hat, hidden)
            xt_hat = torch.add(xt_hat, xt_delta_hat)
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def loss(self, x, burn_in_length, rollout_length):
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
    
    
class ReplayBuffer:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def upload_training_set(self, training_set):
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set
        
    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        episode_starts = np.random.randint(0, (self.episode_length - self.sequence_length)+1, size=batch_size)
        batch_trajs = []
        for i,ep_ind in enumerate(episode_inds):
            start = episode_starts[i]
            end = start + self.sequence_length
            traj_sample = self.buffer[ep_ind,start:end,:]
            assert traj_sample.size(0) == self.sequence_length
            batch_trajs.append(traj_sample)
        trajectories = torch.stack(batch_trajs, dim=0)
        
        return trajectories

""" Tranformer world model adapted from IRIS(https://github.com/eloialonso/iris/blob/main/src/models/world_model.py) """

" Some util functions for the world model "
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self
    

" main transformer block "
class SelfAttention(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        embed_dim = config['embed_dim']
        assert embed_dim % config['num_heads'] == 0
        assert config['attention'] in ['causal', 'window', 'square']
        self.num_heads = config['num_heads']
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self._create_attn_mask()

    def _create_attn_mask(self) -> None:
        mask_type = self.config['attention']
        max_seq_len = self.config['max_seq_len']
        if mask_type == 'causal':    # autoregressive mask, keeps diagonal and below
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        elif mask_type == 'window':
            # slide window of 
            mask = torch.zeros(max_seq_len, max_seq_len)
            for i in range(self.config['max_seq_len']):
                # can attend to previous window_size steps
                mask[i, : min(i + self.config['window_size'] + 1, max_seq_len)] = 1
        elif mask_type == 'square':
            mask = torch.ones(max_seq_len, max_seq_len)
            mask[:, :self.config['square_size']] = 1        
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))        
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

        y = self.resid_drop(self.proj(y))

        return y
    

class Block(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        embed_dim = config['embed_dim']
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x
    

class Transformer(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config["embed_pdrop"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_attn_layers'])])
        self.ln_f = nn.LayerNorm(config['embed_dim'])

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)        
        x = self.drop(sequences)        
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


" World model "
class TransformerIrisWorldModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config        
        self.embedder = self._build_encoder_mlp(self.config['num_encode_layers'])
        if 'pos_encode_aggregation' not in self.config:
            self.pos_emb = nn.Embedding(config['max_seq_len'], config['embed_dim']) 
        else:
            #self.config['pos_encode_aggregation'] == 'concat': 
            assert 'pos_embed_dim' in self.config, \
                "pos_embed_dim must be specified in config if pos_encode_aggregation is concat"
            self.pos_emb = nn.Embedding(config['max_seq_len'], config['pos_embed_dim'])
            self.pos_emb_proj = nn.Linear(config['embed_dim'] + config['pos_embed_dim'], config['embed_dim'])
        
            
        self.transformer = Transformer(config)
        self.decoder = nn.Linear(self.config['embed_dim'], self.config['obs_size'])     #self._build_decoder_mlp(self.config['num_decode_layers'])
        self.apply(init_weights)

    def __repr__(self) -> str:
        return "TranformerIrisWorldModel"
        
    def _build_encoder_mlp(self, num_layers):
        layers = [
            nn.Linear(self.config['obs_size'], self.config['embed_dim']),
            nn.LayerNorm(self.config['embed_dim']),
            nn.GELU()]
        for _ in range(1, num_layers-1):
            layers.append(nn.Linear(self.config['embed_dim'], self.config['embed_dim']))
            layers.append(nn.GELU())
        layers += [
            nn.Linear(self.config['embed_dim'], self.config['embed_dim']),
            nn.LayerNorm(self.config['embed_dim'])]
        return nn.Sequential(*layers)
    
    def _build_decoder_mlp(self, num_layers):
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Linear(self.config['embed_dim'], self.config['embed_dim']))
            layers.append(nn.GELU())
        layers.append(nn.Linear(self.config['embed_dim'], self.config['obs_size']))
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.FloatTensor, past_keys_values: Optional[KeysValues] = None):
        num_steps = obs.size(1)  # (B, T)
        assert num_steps <= self.config['max_seq_len'], "num_steps should be less than or equal to max_seq_len"
        prev_steps = 0 if past_keys_values is None else past_keys_values.size        
        embds = self.embedder(obs)
        self.embds = embds
        self.pos_embds = self.pos_emb(prev_steps + torch.arange(num_steps, device=obs.device))
        if 'pos_encode_aggregation' not in self.config:
            seq_emdbs = embds + self.pos_embds        
        else:
            #if self.config['pos_encode_aggregation'] == 'concat':            
            pos_embds = self.pos_embds.unsqueeze(0).expand(embds.size(0), -1, -1) # (B, T, E)
            seq_emdbs = self.pos_emb_proj(torch.cat([embds, pos_embds], dim=-1))        
        x = self.transformer(seq_emdbs, past_keys_values)        
        output_observations = self.decoder(x)
        return (x, output_observations)
    
    def loss(self, obs: torch.LongTensor) -> torch.FloatTensor:
        # somehow new data has dtype float64
        if obs.dtype == torch.float64:
            obs = obs.float()
        if self.config['attention'] == 'causal':
            inputs = obs[:, :-1]
            labels = obs[:, 1:]
        elif self.config['attention'] == 'window':
            # model can see up to T-window_size steps and predict the window_size steps
            assert self.config['window_size'] is not None, "window_size should be specified"
            assert self.config['window_size'] < obs.size(1), \
                f"window_size should be at less than {obs.size(1)}"
            inputs = obs[:, :-self.config['window_size']]
            labels = obs[:, self.config['window_size']:]
        elif self.config['attention'] == 'square':
            assert self.config['square_size'] >= obs.size(1) / 2, \
                f"square_size should be at least {obs.size(1)/2}"
            inputs = obs[:, : self.config['square_size']]
            labels = obs[:, self.config['square_size'] :]
                
        x, output_observations = self(inputs)
        # sqaure mask input and labels won't be the same size
        if self.config['attention'] == 'square':
            output_observations = output_observations[:, :labels.size(1)]

        outputs = rearrange(output_observations, 'b t o -> (b t) o')        
        labels = rearrange(labels, 'b t o -> (b t) o')        
        loss = F.mse_loss(labels, outputs)
        return loss
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        x_burn_in = x[:, :burn_in_length]
        x_rollout = []
        for i in range(rollout_length):          
           _, output_observations = self(x_burn_in)
           x_burn_in = output_observations
           x_rollout.append(output_observations[:, -1])
           #print(f"rollout step {i}, mse is {F.mse_loss(x[:, burn_in_length + i], output_observations[:, -1])}")
        x_rollout = torch.stack(x_rollout, dim=1)        
        return x_rollout