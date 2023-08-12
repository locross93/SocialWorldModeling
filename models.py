# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:09:25 2022

@author: locro
"""

from collections import namedtuple
from typing import Tuple, Dict
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as td


class MlpAutoencoder(torch.nn.Module):
    def __init__(self, config):
        super(MlpAutoencoder, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size  = config['hidden_size']
        self.latent_size = config['latent_size']
        
        # Define the recognition model (encoder or q) part
        self.enc_fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.enc_fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.enc_fc3 = torch.nn.Linear(self.hidden_size, self.latent_size)
        
        # Define the generative model (decoder or p) part
        self.dec_fc1 = torch.nn.Linear(self.latent_size, self.hidden_size)
        self.dec_fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dec_fc3 = torch.nn.Linear(self.hidden_size, self.input_size)
        
    def encoder(self, x):
        hidden = F.relu(self.enc_fc1(x))
        hidden = F.relu(self.enc_fc2(hidden))
        z = self.enc_fc3(hidden)
        return z
    
    def decoder(self, z):
        s = F.relu(self.dec_fc1(z))
        s = F.relu(self.dec_fc2(s))
        mu_x = self.dec_fc3(s)
        return mu_x
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def l2(self, x):
        x_hat = self.forward(x)
        l2 = ((x - x_hat)**2).sum()
        return l2
    
class MlpVae(torch.nn.Module):
    def __init__(self, config):
        super(MlpVae, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size  = config['hidden_size']
        self.latent_size = config['latent_size']
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        
        # Define the recognition model (encoder or q) part
        self.enc_fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.enc_fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.enc_fc_mu = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.enc_fc_sigma = torch.nn.Linear(self.hidden_size, self.latent_size)
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
        # Define the generative model (decoder or p) part
        self.dec_fc1 = torch.nn.Linear(self.latent_size, self.hidden_size)
        self.dec_fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dec_fc3 = torch.nn.Linear(self.hidden_size, self.input_size)
        
    def encoder(self, x):
        hidden = F.relu(self.enc_fc1(x))
        hidden = F.relu(self.enc_fc2(hidden))
        mu = self.enc_fc_mu(hidden)
        # An exponential activation is often added to σ(x) to ensure the result is positive
        sigma = torch.exp(self.enc_fc_sigma(hidden))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
    def decoder(self, z):
        s = F.relu(self.dec_fc1(z))
        s = F.relu(self.dec_fc2(s))
        mu_x = self.dec_fc3(s)
        return mu_x
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def elbo(self, x):
        x_hat = self.forward(x)
        elbo = ((x - x_hat)**2).sum() + self.beta*self.kl
        return elbo
    
    
class LSTM_VAE(torch.nn.Module):
    def __init__(self, config):
        super(LSTM_VAE, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size  = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_layers = config['num_layers']
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.mean = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        
        # Decoder Part
        self.hidden_decoder = self.init_hidden(32)
        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        self.decoder_lstm = torch.nn.LSTM(input_size=(self.latent_size+self.input_size), hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.output = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.input_size)
    
    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)
    
    def encoder(self, x):
    	# TO DO MAKE BATCH SIZE DEPENDENT ON X
        hidden_encoder = self.init_hidden(self.batch_size)
        output_encoder, hidden_encoder = self.encoder_lstm(x, hidden_encoder)
        
        # Extimate the mean and the variance of q(z|x)
        mean = self.mean(hidden_encoder[0])
        log_var = self.log_variance(hidden_encoder[0])
        std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
        
        # Generate a unit gaussian noise.
        seq_len = output_encoder.size(1)
        noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
        
        z = noise * std + mean
        
        self.kl = (std**2 + mean**2 - torch.log(std) - 1/2).sum()

        return z
    
    def decoder(self, z):
        self.hidden_decoder = self.init_hidden(self.batch_size) 
        
        x = self.init_hidden_decoder(z)
        x = torch.reshape(x, (self.batch_size, 1, x.shape[-1]))
        z = torch.reshape(z, (self.batch_size, 1, z.shape[-1]))
        
        # Store the outputs of the model at each time step
        outputs = []
        
        # Iterate over the time steps
        for t in range(self.num_steps):
            # Concatenate the input features and latent features
            combined_features = torch.cat((x, z), dim=2)
            output_decoder, self.hidden_decoder = self.decoder_lstm(combined_features, self.hidden_decoder)
            
            # Pass the final LSTM output through the linear layer to produce the output
            output = self.output(output_decoder)
            outputs.append(torch.reshape(output, (self.batch_size, output.shape[-1])))

            # Use the output as the input for the next time step
            x = output
        x_hat = torch.stack(outputs, dim=1)
        
        return x_hat
    
    def forward(self, x):
        self.batch_size = x.size(0)
        self.num_steps = x.size(1)
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder(z)
        
        return x_hat
    
    def elbo(self, x):
        x_hat = self.forward(x)
        elbo = ((x - x_hat)**2).sum() + self.beta*self.kl

        return elbo
    
    
class LSTM_VAE_delta(torch.nn.Module):
    def __init__(self, config):
        super(LSTM_VAE_delta, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size  = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_layers = config['num_layers']
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.mean = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        
        # Decoder Part
        self.hidden_decoder = self.init_hidden(32)
        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        self.decoder_lstm = torch.nn.LSTM(input_size=(self.latent_size+self.input_size), hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.decoder_fc = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.hidden_size)
        self.output = torch.nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
    
    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)
    
    def encoder(self, x):
        hidden_encoder = self.init_hidden(self.batch_size)
        output_encoder, hidden_encoder = self.encoder_lstm(x, hidden_encoder)
        
        # Extimate the mean and the variance of q(z|x)
        mean = self.mean(hidden_encoder[0])
        log_var = self.log_variance(hidden_encoder[0])
        std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
        
        # Generate a unit gaussian noise.
        seq_len = output_encoder.size(1)
        noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
        
        z = noise * std + mean
        
        self.kl = (std**2 + mean**2 - torch.log(std) - 1/2).sum()

        return z
    
    def decoder(self, z):
        self.hidden_decoder = self.init_hidden(self.batch_size) 
        
        x = self.init_hidden_decoder(z)
        x = torch.reshape(x, (self.batch_size, 1, x.shape[-1]))
        z = torch.reshape(z, (self.batch_size, 1, z.shape[-1]))
        
        # Store the outputs of the model at each time step
        outputs = []
        
        # Iterate over the time steps
        for t in range(self.num_steps):
            # Concatenate the input features and latent features
            combined_features = torch.cat((x, z), dim=2)
            output_decoder, self.hidden_decoder = self.decoder_lstm(combined_features, self.hidden_decoder)
            
            # Pass the final LSTM output through a linear layer, relu, and another linear layer to produce the output
            output_decoder_thr = F.relu(self.decoder_fc(output_decoder))
            output = self.output(output_decoder_thr)

            # add output of linear layer to previous observation to produce prediction of next observation
            # Use the output as the input for the next time step
            x = torch.add(x, output)
            
            # add to outputs
            outputs.append(torch.reshape(x, (self.batch_size, x.shape[-1])))
            
        x_hat = torch.stack(outputs, dim=1)
        
        return x_hat
    
    def forward(self, x):
        self.batch_size = x.size(0)
        self.num_steps = x.size(1)
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder(z)
        
        return x_hat
    
    def elbo(self, x):
        x_hat = self.forward(x)
        elbo = ((x - x_hat)**2).sum() + self.beta*self.kl

        return elbo
    
    
class TRAJ_VAE(torch.nn.Module):
    def __init__(self, config):
        super(TRAJ_VAE, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size  = config['hidden_size']
        self.latent_size = config['latent_size']
        self.num_layers = config['num_layers']
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        if 'mean_pool' in config:
            self.mean_pool = config['mean_pool']
        else:
            self.mean_pool = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.mean = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        
        # Decoder Part
        self.hidden_decoder = self.init_hidden(32)
        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        self.decoder_lstm = torch.nn.LSTM(input_size=(self.latent_size+self.input_size), hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.output = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.input_size)
    
    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)
    
    def encoder(self, x):
        self.batch_size = x.size()[0]
        self.num_steps = x.size()[1]
        hidden_encoder = self.init_hidden(self.batch_size) 
        output_encoder, hidden_encoder = self.encoder_lstm(x, hidden_encoder)
        
        # Extimate the mean and the variance of q(z|x)
        if self.mean_pool:
            hidden_rep = torch.mean(output_encoder, dim=1)
        else:
            hidden_rep = hidden_encoder[0]
        mean = self.mean(hidden_rep)
        log_var = self.log_variance(hidden_rep)
        std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
        
        # Generate a unit gaussian noise.
        seq_len = output_encoder.size(1)
        noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
        
        z = noise * std + mean
        
        self.kl = (std**2 + mean**2 - torch.log(std) - 1/2).sum()

        return z
    
    def decoder(self, z):
        self.hidden_decoder = self.init_hidden(self.batch_size) 
        
        x = self.init_hidden_decoder(z)
        x = torch.reshape(x, (self.batch_size, 1, x.shape[-1]))
        z = torch.reshape(z, (self.batch_size, 1, z.shape[-1]))
        
        # Store the outputs of the model at each time step
        outputs = []
        
        # Iterate over the time steps
        for t in range(self.num_steps):
            # Concatenate the input features and latent features
            combined_features = torch.cat((x, z), dim=2)
            output_decoder, self.hidden_decoder = self.decoder_lstm(combined_features, self.hidden_decoder)
            
            # Pass the final LSTM output through the linear layer to produce the output
            output = self.output(output_decoder)
            outputs.append(torch.reshape(output, (self.batch_size, output.shape[-1])))

            # Use the output as the input for the next time step
            x = output
        x_hat = torch.stack(outputs, dim=1)
        
        return x_hat
    
    def forward(self, x):
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder(z)
        
        return x_hat
    
    def elbo(self, x):
        x_hat = self.forward(x)
        elbo = ((x - x_hat)**2).sum() + self.beta*self.kl

        return elbo
    
    
class SEQ_VAE(torch.nn.Module):
    def __init__(self, config):
        super(SEQ_VAE, self).__init__()
        self.input_size = config['input_size']
        self.deter_size  = config['deter_size']
        self.stoch_size = config['stoch_size']
        self.dec_hidden_size = config['dec_hidden_size']
        self.num_layers = config['num_layers']
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        if 'mean_pool' in config:
            self.mean_pool = config['mean_pool']
        else:
            self.mean_pool = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True, num_layers=self.num_layers)
        self.mean_post = torch.nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        self.log_var_post = torch.nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        
        # Decoder Part
        self.decoder_mlp = torch.nn.Sequential(
            # First layer
            torch.nn.Linear(self.stoch_size, self.dec_hidden_size),
            torch.nn.ReLU(),
            # Second layer
            torch.nn.Linear(self.dec_hidden_size, self.input_size),
            )
    
    def init_d_states(self, batch_size):
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
        z_post = self.init_s_states(self.batch_size)
        
        zs = []
        kls = []
        for t in range(x.size(1)):
            xt = x[:,t,:]
            xt = xt.reshape(xt.size(0), 1, xt.size(1))

            output_encoder, hidden_encoder = self.encoder_lstm(z_post, hidden_encoder)
            # Estimate posterior stochastic state
            post_input = torch.cat((xt, hidden_encoder[0].permute(1, 0, 2)), dim=2)
            mean_post = self.mean_post(post_input)
            log_var_post = self.log_var_post(post_input)
            std_post = torch.exp(0.5 * log_var_post)
            noise_post = torch.randn(self.batch_size, 1, self.stoch_size).to(self.device)
            z_post = noise_post * std_post + mean_post
            zs.append(torch.reshape(z_post, (self.batch_size, z_post.shape[-1])))
            
            # Calculate KL Divergence between prior and posterior
            kl = (std_post**2 + mean_post**2 - torch.log(std_post) - 1/2).sum()
            kls.append(kl)
        
        z = torch.stack(zs, dim=1)
        self.kl = torch.stack(kls, dim=0).sum()

        return z
    
    def forward(self, x):
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder_mlp(z)
        
        return x_hat
    
    def elbo(self, x):
        x_hat = self.forward(x)
        elbo = ((x - x_hat)**2).sum() + self.beta*self.kl

        return elbo
    
RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter']) 
    

class RSSM(nn.Module):
    def __init__(self, config):
        super(RSSM, self).__init__()
        self.input_size = config['input_size']
        self.deter_size  = config['deter_size']
        self.stoch_size = config['stoch_size']
        self.dec_hidden_size = config['dec_hidden_size']
        self.dec_num_layers = config['dec_num_layers']
        if 'dec_dist' in config:
            self.dec_dist = config['dec_dist']
        else:
            self.dec_dist = None
        if 'beta' in config:
            self.beta = config['beta']
        else:
            self.beta = 1.0
        if 'rssm_type' in config:
            self.rssm_type = config['rssm_type']
        else:
            self.rssm_type = 'continuous'
        if 'rnn_type' in config:
            self.rnn_type = config['rnn_type']
        else:
            self.rnn_type = 'GRU'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        if self.rnn_type == 'GRU':
            self.encoder_rnn = nn.GRU(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
        self.mean_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
        self.log_var_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
        self.mean_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        self.log_var_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        
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
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        
        return kl_loss
    
    def get_dist(self, rssm_state):
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit.shape
            logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)
    
    def forward(self, x):
        # Encoder
        prior, post = self.encoder(x)
        
        # Decoder
        x_hat = self.decoder(post)
        
        return x_hat
    
    def elbo(self, x):
        # Encoder
        prior, post = self.encoder(x)
        
        # KL loss
        kl_loss = self.kl_loss(prior, post)
        
        # Decoder
        x_hat = self.decoder(post)
        
        elbo = ((x - x_hat)**2).sum() + self.beta*kl_loss

        return elbo
    
    
class TRAJ_VAE_delta(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1, beta=1):
        super(TRAJ_VAE_delta, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.beta = beta
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.mean = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.latent_size)
        
        # Decoder Part
        self.hidden_decoder = self.init_hidden(32)
        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_size, out_features=self.input_size)
        self.decoder_lstm = torch.nn.LSTM(input_size=(self.latent_size+self.input_size), hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.decoder_fc = torch.nn.Linear(in_features=self.hidden_size * self.num_layers, out_features=self.hidden_size)
        self.output = torch.nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
    
    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)
    
    def encoder(self, x):
        hidden_encoder = self.init_hidden(self.batch_size)
        output_encoder, hidden_encoder = self.encoder_lstm(x, hidden_encoder)
        
        # Extimate the mean and the variance of q(z|x)
        mean = self.mean(hidden_encoder[0])
        log_var = self.log_variance(hidden_encoder[0])
        std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
        
        # Generate a unit gaussian noise.
        noise = torch.randn(self.batch_size, self.latent_size).to(self.device)
        
        z = noise * std + mean
        
        self.kl = (std**2 + mean**2 - torch.log(std) - 1/2).sum()

        return z
    
    def decoder(self, z):
        self.hidden_decoder = self.init_hidden(self.batch_size) 
        
        x = self.init_hidden_decoder(z)
        x = torch.reshape(x, (self.batch_size, 1, x.shape[-1]))
        z = torch.reshape(z, (self.batch_size, 1, z.shape[-1]))
        
        # Store the outputs of the model at each time step
        outputs = []
        x_hats = [torch.reshape(x, (self.batch_size, x.shape[-1]))]
        
        # Iterate over the time steps
        # Don't compute difference for last time step because we can't supervise on it
        for t in range(self.num_steps-1):
            # Concatenate the input features and latent features
            combined_features = torch.cat((x, z), dim=2)
            output_decoder, self.hidden_decoder = self.decoder_lstm(combined_features, self.hidden_decoder)
            
            # Pass the final LSTM output through a linear layer, relu, and another linear layer to produce the output
            output_decoder_thr = F.relu(self.decoder_fc(output_decoder))
            output = self.output(output_decoder_thr)
            
            # add to outputs
            outputs.append(torch.reshape(output, (self.batch_size, output.shape[-1])))

            # add output of linear layer to previous observation to produce prediction of next observation
            # Use the output as the input for the next time step
            x = torch.add(x, output)
            x_hats.append(torch.reshape(x, (self.batch_size, x.shape[-1])))
            
        x_delta_hat = torch.stack(outputs, dim=1)
        x_hat = torch.stack(x_hats, dim=1)
        
        return x_delta_hat, x_hat
    
    def forward(self, x):
        self.batch_size = x.size(0)
        self.num_steps = x.size(1)
        # Encoder
        z = self.encoder(x)
        
        # Decoder
        x_delta_hat = self.decoder(z)
        
        return x_delta_hat
    
    def elbo(self, x):
        # compute the difference between each feature in consecutive timepoints
        x_delta = x[:, 1:, :] - x[:, :-1, :]
        
        # model outputs the predicted difference
        x_delta_hat, x_hat = self.forward(x)
        
        # supervise on predicting the difference with kl reg
        elbo = ((x_delta - x_delta_hat)**2).sum() + self.beta*self.kl

        return elbo
    

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
    
    
class RSSM_Delta(nn.Module):
    def __init__(self, config): 
        super(RSSM_Delta, self).__init__()
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
            
        if self.rssm_type == 'discrete':
            self.class_size = config['class_size']
            self.category_size = config['category_size']
            self.stoch_size = self.class_size*self.category_size
            self.fc_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.fc_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        elif self.rssm_type == 'continuous':
            self.stoch_size = config['stoch_size']
            self.mean_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.log_var_prior = nn.Linear(in_features=self.deter_size, out_features=self.stoch_size)
            self.mean_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
            self.log_var_post = nn.Linear(in_features=self.deter_size+self.input_size, out_features=self.stoch_size)
        
        # Encoder Part
        if self.rnn_type == 'GRU':
            self.encoder_rnn = nn.GRU(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(input_size=self.stoch_size, hidden_size=self.deter_size, batch_first=True)
            
        # Decoder Part
        if self.dec_dist == 'normal' or self.dec_dist == 'laplace':
            self.output_size = self.input_size * 2
        elif self.dec_dist == None:
            self.output_size = self.input_size        
        self.decoder_mlp = self.build_mlp(self.dec_num_layers, self.stoch_size+self.deter_size, self.dec_hidden_size, self.output_size, nn.ELU)
        
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

            # use z for input of next rnn iteration, resized for (B, T, Z)            
            z_post_in = z_post.reshape(z_post.size(0), 1, z_post.size(1))
        
        prior = self.rssm_stack_states(priors, dim=1)
        post = self.rssm_stack_states(posteriors, dim=1)
        
        return prior, post
    
    def decoder(self, posterior):
        rssm_state = torch.cat((posterior.deter, posterior.stoch), dim=-1)
        dist_inputs = self.decoder_mlp(rssm_state)
        # remove the last timepoints prediction bc we don't have a true supervision signal for it
        dist_inputs = dist_inputs[:, :-1, :]
        if self.dec_dist == 'normal':
            #return td.independent.Independent(td.Normal(dist_inputs, 1), len(self._output_shape))
            output_loc, output_logvar = torch.chunk(dist_inputs, 2, dim=-1)
            output_std = output_logvar.exp().pow(0.5)         # logvar to std
            return td.independent.Independent(td.Normal(output_loc, output_std), 1)
        elif self.dec_dist == 'laplace':
            output_loc, output_logvar = torch.chunk(dist_inputs, 2, dim=-1)
            output_scale = output_logvar.exp().pow(0.5)       # logvar to std, analogy may not apply to laplace
            #output_loc, output_scale = torch.chunk(dist_inputs, 2, dim=-1)
            return td.laplace.Laplace(output_loc, output_scale)
        elif self.dec_dist == None:
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
        
    def supervised_loss(self, x_delta, x_delta_hat):
        if self.dec_dist == 'normal' or self.dec_dist == 'laplace':
            # x_delta_hat is a distribution, compute likelihood of data given dist
            sup_loss = -torch.mean(x_delta_hat.log_prob(x_delta))
        elif self.dec_dist == None:
            sup_loss = ((x_delta - x_delta_hat)**2).sum()
        return sup_loss
        
    def kl_loss(self, prior, posterior):
        prior_dist = self.get_dist(prior)
        post_dist = self.get_dist(posterior)
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
    
    def forward(self, x):
        # Encoder
        prior, post = self.encoder(x)

        # Decoder
        x_delta_hat = self.decoder(post)
        if self.dec_dist != None:
            # decoder outputs a distribution, sample from it to get reconstructions
            x_delta_hat = x_delta_hat.sample()
        x_hat = x.detach().clone()
        #x_hat[:, 1:, :] = x[:, :-1, :] + x_delta_hat[:, :-1, :]
        x_hat[:, 1:, :] = x[:, :-1, :] + x_delta_hat
        
        return x_delta_hat, x_hat
    
    def loss(self, x):
        # compute the difference between each feature in consecutive timepoints
        x_delta = x[:, 1:, :] - x[:, :-1, :]
        
        # Encoder
        prior, post = self.encoder(x)
        
        # KL loss
        kl_loss = self.kl_loss(prior, post)
        
        # Decoder - predict the difference between the current state and the next
        x_delta_hat = self.decoder(post)
        # # remove the last timepoints prediction bc we don't have a true supervision signal for it
        # if self.dec_dist == None:
        #     x_delta_hat = x_delta_hat[:, :-1, :]

        # supervise on predicting the difference with kl reg
        sup_loss = self.supervised_loss(x_delta, x_delta_hat)
        elbo = sup_loss + self.beta*kl_loss

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
        assert len(pred_outputs) != 0
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def loss(self, x, burn_in_length, rollout_length):
        loss_fn = torch.nn.MSELoss()
        
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = loss_fn(x_supervise, x_hat)
        
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


class MultistepPredictor4D(nn.Module):
    def __init__(self, config):
        super(MultistepPredictor4D, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.num_stack = config['num_stack']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.input_embed_size = config['input_embed_size']
        self.input_embed_layers = config['input_embed_layers']
        self.rnn_input_size = self.input_embed_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.input_embed = self.build_mlp(self.input_embed_layers, self.input_size*self.num_stack, self.input_embed_size, self.input_embed_size, nn.ReLU)
        
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
            # Construct new frame-stacked data: discard the oldest frame and append the newest prediction [t, t-1, t-2, t-3]
            new_input = torch.cat([xt_hat, x[:, (burn_in_length-1+t), :self.input_size*(self.num_stack-1)].unsqueeze(1)], dim=2)
            xt_hat, hidden = self.forward(new_input, hidden)
        assert len(pred_outputs) != 0
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def loss(self, x, burn_in_length, rollout_length):
        loss_fn = torch.nn.MSELoss()
        
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:self.input_size]
        
        loss = loss_fn(x_supervise, x_hat)
        
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


class ReplayBufferFrameStack:
    def __init__(self, sequence_length, num_stack):
        self.sequence_length = sequence_length
        self.num_stack = num_stack

    def upload_training_set(self, training_set):
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set

    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        episode_starts = np.random.randint(self.num_stack, (self.episode_length - self.sequence_length)+1, size=batch_size)
        
        batch_trajs = []
        for i, ep_ind in enumerate(episode_inds):
            stacked_trajs = []
            for j in range(self.num_stack):
                start = episode_starts[i] - j
                end = start + self.sequence_length
                traj_sample = self.buffer[ep_ind, start:end, :]
                assert traj_sample.size(0) == self.sequence_length
                stacked_trajs.append(traj_sample)
            stacked_traj = torch.cat(stacked_trajs, dim=1)
            batch_trajs.append(stacked_traj)
            
        trajectories = torch.stack(batch_trajs, dim=0)
        
        return trajectories


class ReplayBufferEarly:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def upload_training_set(self, training_set):
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set
        
    # def sample(self, batch_size, random_seed=None):
    #     if random_seed is not None:
    #         np.random.seed(random_seed)
    #     episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
    #     #episode_starts = np.random.randint(0, (self.episode_length - self.sequence_length)+1, size=batch_size)
    #     episode_starts = np.random.randint(0, 100, size=batch_size)
    #     batch_trajs = []
    #     for i,ep_ind in enumerate(episode_inds):
    #         start = episode_starts[i]
    #         end = start + self.sequence_length
    #         traj_sample = self.buffer[ep_ind,start:end,:]
    #         assert traj_sample.size(0) == self.sequence_length
    #         batch_trajs.append(traj_sample)
    #     trajectories = torch.stack(batch_trajs, dim=0)
        
    #     return trajectories
    
    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        # generate probabilities that decrease linearly from start to end of the episode
        p = np.linspace(1, 0, (self.episode_length - self.sequence_length) + 1)
        p = p / p.sum()  # normalize probabilities
        episode_starts = np.random.choice(
            np.arange(0, (self.episode_length - self.sequence_length) + 1), 
            size=batch_size, 
            replace=True, 
            p=p  # use the generated probabilities
        )
        batch_trajs = []
        for i, ep_ind in enumerate(episode_inds):
            start = episode_starts[i]
            end = start + self.sequence_length
            traj_sample = self.buffer[ep_ind, start:end, :]
            assert traj_sample.size(0) == self.sequence_length
            batch_trajs.append(traj_sample)
        trajectories = torch.stack(batch_trajs, dim=0)
    
        return trajectories

    
    
class TransformerMSPredictor(nn.Module):
    def __init__(self, config):
        super(TransformerMSPredictor, self).__init__()
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.num_heads = config['num_heads']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.dropout_p = 0.1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LAYERS
        #self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.embedding = nn.Linear(self.input_size, self.embedding_size)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.embedding_size, dropout_p=self.dropout_p, max_len=5000
        )
        self.transformer = nn.Transformer(
            d_model=self.embedding_size,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout=self.dropout_p,
        )
        self.out = nn.Linear(self.embedding_size, self.input_size)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding_size)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, feat_dim)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
    
    def get_tgt_mask(self, size, mask_type='triangular'):
        if mask_type == 'triangular':
            # Generates a square matrix where the each row allows one step more to be seen
            mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
            mask = mask.float()
            mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
            mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
            
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]
        elif mask_type == 'square':
            # Generates a square matrix where no all elements are masked out
            # THIS DOESN'T WORK, TRANSFORMER SPITS OUT NANS
            mask = torch.full((size,size), float('-inf'))
        
        return mask
    
    def forward_rollout(self, x, context_length, rollout_length):
        src = x[:,:context_length,:]
        # Tensor to hold predictions
        x_hat = torch.zeros(x.shape[0], rollout_length, x.shape[2]).to(x.device)

        for i in range(rollout_length):
            # Use transformer to predict next step, use the last step of context as the tgt
            out = self.forward(src, src[:,-1,:].unsqueeze(1))
            # Permute pred to have batch size first again
            out = out.permute(1, 0, 2)  
            # Append prediction to predictions tensor
            x_hat[:,i,:] = out.squeeze(1)
            # Append prediction to context for next prediction
            src = torch.cat((src, out), dim=1)
        
        return x_hat
    
    def loss(self, x, context_length, rollout_length, mask_type):
        x_hat = self.forward_rollout(x, context_length, rollout_length)  
        t_end = context_length + rollout_length
        x_supervise = x[:,context_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
    
    
class TransformerWorldModel(nn.Module):
    def __init__(self, config):
        super(TransformerWorldModel, self).__init__()
        self.input_size = config['input_size']
        self.embedding_size = config['embedding_size']
        self.num_heads = config['num_heads']
        self.encoder_hidden_size = config['encoder_hidden_size']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.decoder_hidden_size = config['decoder_hidden_size']
        self.dropout_p = 0.1
        self.context_length = config['context_length']
        self.rollout_length = config['rollout_length']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # LAYERS
        #self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.embedding = nn.Linear(self.input_size, self.embedding_size)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.embedding_size, dropout_p=self.dropout_p, max_len=5000
        )
        encoder_layers = nn.TransformerEncoderLayer(self.embedding_size, self.num_heads, self.encoder_hidden_size, self.dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_encoder_layers)
        self.decoder = self.build_mlp(self.num_decoder_layers, self.embedding_size*self.context_length, self.decoder_hidden_size, self.input_size*self.rollout_length, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)
    
    def forward(self, src):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.embedding_size)
        src = self.positional_encoder(src)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer_encoder(src)
        # Permute pred to have batch size first again
        transformer_out = transformer_out.permute(1, 0, 2)
        decoder_input = transformer_out.reshape(transformer_out.shape[0], -1)
        out = self.decoder(decoder_input)
        out = out.reshape(out.shape[0], self.rollout_length, self.input_size)
        
        return out
    
    def forward_rollout_train(self, x):
        src = x[:,:self.context_length,:]
        x_hat = self.forward(src)
        
        return x_hat
    
    def forward_rollout(self, x, context_length, rollout_length):
        sequence_length = context_length + rollout_length
        # only input obs from x up to context_length
        src = x[:,:context_length,:]

        # generate predictions until 
        while src.size(1) < sequence_length:
            if src.size(1) < self.context_length:
                # if context is less than the model's context length, pad the beginning with zeros
                zero_padding = torch.zeros(src.size(0), self.context_length - src.size(1), src.size(2)).to(x.device)
                src_temp = torch.cat((zero_padding, src), dim=1)
                out = self.forward(src_temp)
            elif src.size(1) > self.context_length:
                # use the last 50 steps
                out = self.forward(src[:,-self.context_length:,:])
            else:
                out = self.forward(src)
            # Append prediction to context for next prediction
            src = torch.cat((src, out), dim=1)
        x_hat = src[:,context_length:sequence_length,:]
        
        assert x_hat.size(1) == rollout_length
        
        return x_hat
    
    def loss(self, x):
        x_hat = self.forward_rollout_train(x)  
        t_end = self.context_length + self.rollout_length
        x_supervise = x[:,self.context_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
        
        
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
    
    
class AdversarialCritic(nn.Module):
    def __init__(self, config):
        super(AdversarialCritic, self).__init__()
        self.input_size = config['input_size']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'normalize' in config:
            self.normalize = config['normalize']
        else:
            self.normalize = False
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, 1, nn.ReLU)
        if self.normalize:
            self.sigmoid = nn.Sigmoid()
        
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)
    
    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
        state_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
        return (hidden_cell, state_cell)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        state_value = self.mlp(out)
        if self.normalize:
            state_value = self.sigmoid(state_value)
        return state_value, hidden
    
    
class MSPredictorEndStateContext(nn.Module):
    def __init__(self, config):
        super(MSPredictorEndStateContext, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # create two linear embeddings, one for time varying input, one for predicted end state by separate model
        self.input_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size//2)
        self.end_state_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size//2)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.input_size, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x, end_state_hat, hidden):
        input_embed = self.input_embed(x)
        # copy predicted end state for each time step
        if len(end_state_hat.size()) == 2:
            end_state_hat = end_state_hat.unsqueeze(1)
            end_state_hat = end_state_hat.repeat_interleave(x.size(1), dim=1)
        end_state_embed = self.end_state_embed(end_state_hat)
        rnn_input = torch.cat((input_embed, end_state_embed), dim=2)
        out, hidden = self.rnn(rnn_input, hidden)
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
    
    def forward_rollout(self, x, end_state_hat, burn_in_length, rollout_length):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x[:,:burn_in_length,:], end_state_hat, hidden)
        # last output is next input x
        xt_hat = output[:,-1,:].unsqueeze(1)
        pred_outputs = []
        for t in range(rollout_length):
            pred_outputs.append(xt_hat.squeeze(1))
            # output is next input x
            xt_hat, hidden = self.forward(xt_hat, end_state_hat, hidden)
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def supervised_loss(self, x, end_state_hat, burn_in_length, rollout_length):
        x_hat = self.forward_rollout(x, end_state_hat, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
    
    
class EndStatePredictor(nn.Module):
    def __init__(self, config):
        super(EndStatePredictor, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
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
    
    def supervised_loss(self, x, end_state):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x, hidden)
        # last output is prediction with entire burn in sequence
        end_state_hat = output[:,-1,:]
        assert end_state.size() == end_state_hat.size()
        
        loss = ((end_state - end_state_hat)**2).sum()
        
        return loss, end_state_hat
    
    
class EndStateHorizonPredictor(nn.Module):
    def __init__(self, config):
        super(EndStateHorizonPredictor, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.horizon_loss_weight = 1e2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp_end_state = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.input_size, nn.ReLU, out_sigmoid=False)
        self.mlp_end_horizon = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, 1, nn.ReLU, out_sigmoid=True) 
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation, out_sigmoid):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        if out_sigmoid:
            # normalize output between 0 and 1
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, x, hidden):
        rnn_out, hidden = self.rnn(x, hidden)
        out_end_state = self.mlp_end_state(rnn_out)
        out_end_horizon = self.mlp_end_horizon(rnn_out)
        return out_end_state, out_end_horizon, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return hidden_cell
        elif self.rnn_type == 'LSTM':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            state_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return (hidden_cell, state_cell)
    
    def supervised_loss(self, x, end_state, end_horizon):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out_end_state, out_end_horizon, hidden = self.forward(x, hidden)
        # last output is prediction with entire burn in sequence
        end_state_hat = out_end_state[:,-1,:]
        assert end_state.size() == end_state_hat.size()
        end_horizon_hat = out_end_horizon[:,-1,:].squeeze(1)
        assert end_horizon.size() == end_horizon_hat.size()
        
        end_state_loss = ((end_state - end_state_hat)**2).sum()
        horizon_loss = self.horizon_loss_weight * ((end_horizon - end_horizon_hat)**2).sum()
        loss = end_state_loss + horizon_loss
        
        return loss, end_state_loss, horizon_loss, end_state_hat, end_horizon_hat
    
    
class MPEndStateContextInputTime(nn.Module):
    def __init__(self, config):
        super(MPEndStateContextInputTime, self).__init__()
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.time_input_type = config['time_input_type'] # scalar or ohe
        self.episode_length = 300
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # create two linear embeddings, one for time varying input, one for predicted end state by separate model
        self.input_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size//2)
        self.end_state_embed = nn.Linear(in_features=self.output_size, out_features=self.mlp_hidden_size//2)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.output_size, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x, end_state_hat, hidden):
        input_embed = self.input_embed(x)
        # copy predicted end state for each time step
        if len(end_state_hat.size()) == 2:
            end_state_hat = end_state_hat.unsqueeze(1)
            end_state_hat = end_state_hat.repeat_interleave(x.size(1), dim=1)
        end_state_embed = self.end_state_embed(end_state_hat)
        rnn_input = torch.cat((input_embed, end_state_embed), dim=2)
        out, hidden = self.rnn(rnn_input, hidden)
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
    
    def forward_rollout(self, x, end_state_hat, burn_in_length, rollout_length):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x[:,:burn_in_length,:], end_state_hat, hidden)
        # last output is next input x
        xt_hat = output[:,-1,:].unsqueeze(1)
        # get time idx of step = burn_in_length, to create ohe for pred_outputs
        last_time_idx = torch.where(x[:,burn_in_length,self.output_size:])[1]
        pred_outputs = []
        for t in range(rollout_length):
            pred_outputs.append(xt_hat.squeeze(1))
            # output is next input x
            if self.time_input_type == 'ohe':
                # add one-hot encoding of time information to input
                assert not torch.any(last_time_idx >= self.episode_length), "Invalid index for time_step_input"
                time_step_input = F.one_hot(last_time_idx, num_classes=self.episode_length).unsqueeze(1).to(self.device)
            elif self.time_input_type == 'scalar':
                # min max normalize
                time_step_input = (last_time_idx - 0) / (self.episode_length - 1)
                # rescale to -1 and 1
                time_step_input = time_step_input * 2 - 1
                # add two dimensions for concatenation
                time_step_input = time_step_input.unsqueeze(1)
                time_step_input = time_step_input.unsqueeze(1)
            model_input = torch.cat([xt_hat, time_step_input], dim=2) 
            xt_hat, hidden = self.forward(model_input, end_state_hat, hidden)
            last_time_idx = last_time_idx + 1
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def supervised_loss(self, x, end_state_hat, burn_in_length, rollout_length):
        x_hat = self.forward_rollout(x, end_state_hat, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:self.output_size]
        
        loss = ((x_supervise - x_hat)**2).sum()
        
        return loss
    
    
class EndStatePredictorInputTime(nn.Module):
    def __init__(self, config):
        super(EndStatePredictorInputTime, self).__init__()
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # create linear embedding of input that has separate spatial features and temporal one hot encoding
        self.input_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.output_size, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x, hidden):
        input_embed = self.input_embed(x)
        out, hidden = self.rnn(input_embed, hidden)
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
    
    def supervised_loss(self, x, end_state):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x, hidden)
        # last output is prediction with entire burn in sequence
        end_state_hat = output[:,-1,:]
        assert end_state.size() == end_state_hat.size()
        
        loss = ((end_state - end_state_hat)**2).sum()
        
        return loss, end_state_hat
    
    
class RND(nn.Module):
    def __init__(self, config):
        super(RND, self).__init__()
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        
        self.target = self.build_mlp(self.num_layers, self.input_size, self.hidden_size, self.output_size, nn.ReLU)
        self.predictor = self.build_mlp(self.num_layers, self.input_size, self.hidden_size, self.output_size, nn.ReLU)
        self.target.eval()
        
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x):
        return self.target(x), self.predictor(x)
    
    def target_loss(self, x):
        target_out, pred_out = self.forward(x)
        loss = ((target_out - pred_out)**2).sum()
        
        return loss



""" Tranformer world model adapted from IRIS(https://github.com/eloialonso/iris/blob/main/src/models/world_model.py) """
import math
from typing import Optional

from einops import rearrange
from kv_caching import KeysValues, KVCache

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
    

class ReplayBufferEvents:
    def __init__(self, burn_in_length, rollout_length, training_set, event_inds):
        self.burn_in_length = burn_in_length
        self.rollout_length = rollout_length
        self.sequence_length = burn_in_length + rollout_length
        self.event_inds = event_inds
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set
        
    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        episode_starts = np.random.randint(0, (self.episode_length - self.sequence_length)+1, size=batch_size)
        batch_trajs = []
        batch_event_states = []
        batch_event_horizons = []
        for i,ep_ind in enumerate(episode_inds):
            start = episode_starts[i]
            end = start + self.sequence_length
            traj_sample = self.buffer[ep_ind,start:end,:]
            assert traj_sample.size(0) == self.sequence_length
            batch_trajs.append(traj_sample)
            # get closest future event state in that trajectory from burn in state + 10 steps
            traj_event_inds = np.array(self.event_inds[ep_ind])
            burn_in_ind = start + (self.burn_in_length - 1) + 5
            # get the minimum between burn_in_ind and 298
            burn_in_ind = min(burn_in_ind, self.episode_length - 2)
            closest_event_ind = np.min(traj_event_inds[np.where(traj_event_inds > burn_in_ind)[0]])
            # get event horizon from end state
            event_horizon = closest_event_ind - burn_in_ind
            # min-max normalize event horizon, with 0 equaling now, 1 the entire episode length
            event_horizon = (event_horizon - 1) / ((self.episode_length - self.burn_in_length) - 1)
            event_state = self.buffer[ep_ind,closest_event_ind,:]
            batch_event_states.append(event_state)
            batch_event_horizons.append(event_horizon)
        trajectories = torch.stack(batch_trajs, dim=0)
        event_states = torch.stack(batch_event_states, dim=0)
        event_horizons = torch.tensor(batch_event_horizons)
        
        return trajectories, event_states, event_horizons


class ReplayBufferEndState:
    def __init__(self, burn_in_length, rollout_length, training_set, end_traj=False):
        self.burn_in_length = burn_in_length
        self.rollout_length = rollout_length
        self.sequence_length = burn_in_length + rollout_length
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set
        # use end of trial or end of trajectory as end state
        self.end_traj = end_traj
        
    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        episode_starts = np.random.randint(0, (self.episode_length - self.sequence_length)+1, size=batch_size)
        batch_trajs = []
        batch_end_states = []
        batch_end_horizons = []
        for i,ep_ind in enumerate(episode_inds):
            start = episode_starts[i]
            end = start + self.sequence_length
            traj_sample = self.buffer[ep_ind,start:end,:]
            assert traj_sample.size(0) == self.sequence_length
            batch_trajs.append(traj_sample)
            # get last state in that trajectory/trial
            if self.end_traj:
                end_state = self.buffer[ep_ind,end-1,:]
                # end horizon will be fixed at rollout_length
                end_horizon = 1.0
            else:
                end_state = self.buffer[ep_ind,-1,:]
                # how far away is the end state, with 0 equaling now, 1 the entire episode length
                end_horizon = float((self.episode_length - (start + self.burn_in_length)) / (self.episode_length - self.burn_in_length))
            batch_end_states.append(end_state)
            batch_end_horizons.append(end_horizon)
        trajectories = torch.stack(batch_trajs, dim=0)
        end_states = torch.stack(batch_end_states, dim=0)
        end_horizons = torch.tensor(batch_end_horizons)
        
        return trajectories, end_states, end_horizons


class ReplayBufferGTEvents:
    def __init__(self, burn_in_length, rollout_length, training_set, event_inds):
        self.burn_in_length = burn_in_length
        self.rollout_length = rollout_length
        self.sequence_length = burn_in_length + rollout_length
        self.event_inds = event_inds
        self.buffer_size = training_set.size(0)
        self.episode_length = training_set.size(1)
        self.buffer = training_set
        
    def sample(self, batch_size, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        episode_inds = np.random.choice(self.buffer_size, batch_size, replace=False)
        episode_starts = np.random.randint(0, (self.episode_length - self.sequence_length)+1, size=batch_size)
        batch_trajs = []
        batch_event_states = []
        batch_event_horizons = []
        for i,ep_ind in enumerate(episode_inds):
            start = episode_starts[i]
            end = start + self.sequence_length
            traj_sample = self.buffer[ep_ind,start:end,:]
            assert traj_sample.size(0) == self.sequence_length
            batch_trajs.append(traj_sample)
            # get closest future event state in that trajectory from burn in state + 10 steps
            traj_event_inds = np.array(self.event_inds[ep_ind])
            burn_in_ind = start + (self.burn_in_length - 1) + 5
            # get the minimum between burn_in_ind and 298
            burn_in_ind = min(burn_in_ind, self.episode_length - 2)
            next_events_inds = traj_event_inds[np.where(traj_event_inds > burn_in_ind)[0]]
            if len(next_events_inds) == 0 or np.min(next_events_inds) > (self.episode_length - 1):
                # if no events in trial or after burn in, include current state as context
                # also discard rare trials where event happens at ind 300
                closest_event_ind = end - 1
                event_horizon = 0.0
            else:
                closest_event_ind = np.min(next_events_inds)
                # get event horizon from end state
                event_horizon = closest_event_ind - burn_in_ind
                # min-max normalize event horizon, with 0 equaling now, 1 the entire episode length
                event_horizon = float((event_horizon - 1) / ((self.episode_length - self.burn_in_length) - 1))
            event_state = self.buffer[ep_ind,closest_event_ind,:]
            batch_event_states.append(event_state)
            batch_event_horizons.append(event_horizon)
        trajectories = torch.stack(batch_trajs, dim=0)
        event_states = torch.stack(batch_event_states, dim=0)
        event_horizons = torch.tensor(batch_event_horizons)
        
        return trajectories, event_states, event_horizons
    

class EventPredictor(nn.Module):
    def __init__(self, config):
        super(EventPredictor, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        if 'dropout' in config:
            self.dropout = config['dropout']
        else:
            self.dropout = 0
        if 'pred_delta' in config:
            self.pred_delta = config['pred_delta']
        else:
            self.pred_delta = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, dropout=self.dropout, batch_first=True)
        self.event_decoder = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.input_size, nn.ReLU, out_sigmoid=False)
        if config['predict_horizon']:
            self.predict_horizon = True
            self.horizon_loss_weight = config['horizon_loss_weight']
            self.event_horizon_decoder = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, 1, nn.ReLU, out_sigmoid=True) 
        else:
            self.predict_horizon = False

    def build_mlp(self, num_layers, input_size, node_size, output_size, activation, out_sigmoid):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        if out_sigmoid:
            # normalize output between 0 and 1
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, x, hidden):
        rnn_out, hidden = self.rnn(x, hidden)
        out_event_state = self.event_decoder(rnn_out)
        if self.predict_horizon:
            out_event_horizon = self.event_horizon_decoder(rnn_out)
            return out_event_state, out_event_horizon, hidden
        else:
            return out_event_state, hidden

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return hidden_cell
        elif self.rnn_type == 'LSTM':
            hidden_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            state_cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size).to(self.device)
            return (hidden_cell, state_cell)
    
    def supervised_loss(self, x, event_state, event_horizon=None):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        if self.pred_delta:
            # predict delta between current and next event state
            event_state = event_state - x[:,:-1,:]
        if self.predict_horizon:
            out_event_state, out_event_horizon, hidden = self.forward(x, hidden)
        else:
            out_event_state, hidden = self.forward(x, hidden)

        # last output is prediction with entire burn in sequence
        event_hat = out_event_state[:,-1,:]
        try:
            assert event_state.size() == event_hat.size()
        except AssertionError as e:
            print(f"AssertionError caught: {e}")
            import pdb; pdb.set_trace()
        #assert event_state.size() == event_hat.size()
        event_loss = ((event_state - event_hat)**2).mean()

        if self.predict_horizon:
            event_horizon_hat = out_event_horizon[:,-1,:].squeeze(1)
            assert event_horizon.size() == event_horizon_hat.size()
            horizon_loss = self.horizon_loss_weight * ((event_horizon - event_horizon_hat)**2).mean()
            loss = event_loss + horizon_loss
            return loss, event_loss, horizon_loss, event_hat, event_horizon_hat
        else:
            loss = event_loss
            return loss, None, None, event_hat, None
        

class MSPredictorEventContext(nn.Module):
    def __init__(self, config):
        super(MSPredictorEventContext, self).__init__()
        self.input_size = config['input_size']
        self.rnn_type = config['rnn_type']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.num_rnn_layers = config['num_rnn_layers']
        self.mlp_hidden_size = config['mlp_hidden_size']
        if 'mlp_num_layers' in config:
            self.num_mlp_layers = config['num_mlp_layers']
        else:
            self.num_mlp_layers = 2
        if 'dropout' in config:
            self.dropout = config['dropout']
        else:
            self.dropout = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # create two linear embeddings, one for time varying input, one for predicted next event state by separate model
        self.input_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size//2)
        if config['input_pred_horizon']:
            self.input_pred_horizon = True
            # add one to input if inputting predicted horizon
            self.event_embed = nn.Linear(in_features=self.input_size+1, out_features=self.mlp_hidden_size//2)
        else:
            self.input_pred_horizon = False
            self.event_embed = nn.Linear(in_features=self.input_size, out_features=self.mlp_hidden_size//2)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.mlp_hidden_size, self.rnn_hidden_size, num_layers=self.num_rnn_layers, dropout=self.dropout, batch_first=True)
        self.mlp = self.build_mlp(self.num_mlp_layers, self.rnn_hidden_size, self.mlp_hidden_size, self.input_size, nn.ReLU)
    
    def build_mlp(self, num_layers, input_size, node_size, output_size, activation):
        model = [nn.Linear(input_size, node_size)]
        model += [activation()]
        for i in range(num_layers-1):
            model += [nn.Linear(node_size, node_size)]
            model += [activation()]
        model += [nn.Linear(node_size, output_size)]
        return nn.Sequential(*model)

    def forward(self, x, event_hat, hidden):
        input_embed = self.input_embed(x)
        # copy predicted end state for each time step
        if len(event_hat.size()) == 2:
            event_hat = event_hat.unsqueeze(1)
            event_hat = event_hat.repeat_interleave(x.size(1), dim=1)
        event_embed = self.event_embed(event_hat)
        rnn_input = torch.cat((input_embed, event_embed), dim=2)
        out, hidden = self.rnn(rnn_input, hidden)
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
    
    def forward_rollout(self, x, event_hat, burn_in_length, rollout_length):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.forward(x[:,:burn_in_length,:], event_hat, hidden)
        # last output is next input x
        xt_hat = output[:,-1,:].unsqueeze(1)
        pred_outputs = []
        for t in range(rollout_length):
            pred_outputs.append(xt_hat.squeeze(1))
            # output is next input x
            xt_hat, hidden = self.forward(xt_hat, event_hat, hidden)
        x_hat = torch.stack(pred_outputs, dim=1)
        
        return x_hat
    
    def supervised_loss(self, x, event_hat, burn_in_length, rollout_length):
        x_hat = self.forward_rollout(x, event_hat, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = ((x_supervise - x_hat)**2).mean()
        
        return loss


class EventModel:
    def __init__(self, ep_config, mp_config):
        self.ep_model = EventPredictor(ep_config)
        self.mp_model = MSPredictorEventContext(mp_config)
        
    def load_weights(self, ep_weights_path, mp_weights_path):
        # Load the weights of the two models
        self.ep_model.load_state_dict(torch.load(ep_weights_path))
        self.mp_model.load_state_dict(torch.load(mp_weights_path))
        
    def eval(self):
        self.mp_model.eval()
        self.ep_model.eval()

    def predict_next_event(self, x):
        # Call EventPredictor's forward function with burn_in steps - x should be burn_in length
        batch_size = x.size(0)
        hidden = self.ep_model.init_hidden(batch_size)
        if self.ep_model.predict_horizon:
            out_event_state, out_event_horizon, hidden = self.ep_model.forward(x, hidden)
            # last output is prediction with entire burn in sequence
            event_hat = out_event_state[:,-1,:]
            event_horizon_hat = out_event_horizon[:,-1,:].squeeze(1)
            return event_hat, event_horizon_hat
        else:
            out_event_state, hidden = self.ep_model.forward(x, hidden)
            # last output is prediction with entire burn in sequence
            event_hat = out_event_state[:,-1,:]
            return event_hat
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        if self.ep_model.predict_horizon:
            self.event_hat, self.event_horizon_hat = self.predict_next_event(x[:,:burn_in_length,:])
        else:
           self.event_hat = self.predict_next_event(x[:,:burn_in_length,:])

        # condition ms predictor on predicted next event
        if self.mp_model.input_pred_horizon:
            # concatenate event_hat and event_horizon_hat
            self.event_hat = torch.cat([self.event_hat, self.event_horizon_hat.unsqueeze(1)], dim=-1)
        # Call MSPredictorEventContext's forward_rollout with event_hat
        x_hat = self.mp_model.forward_rollout(x, self.event_hat, burn_in_length, rollout_length)

        return x_hat

    def loss(self, x, burn_in_length, rollout_length):
        loss_fn = torch.nn.MSELoss()
        
        x_hat = self.forward_rollout(x, burn_in_length, rollout_length)
        t_end = burn_in_length + rollout_length
        x_supervise = x[:,burn_in_length:t_end,:]
        
        loss = loss_fn(x_supervise, x_hat)
        
        return loss