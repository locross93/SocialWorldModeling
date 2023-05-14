# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:09:25 2022

@author: locro
"""

from collections import namedtuple
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as td
import pdb

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
        #pdb.set_trace()
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
            #pdb.set_trace()
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
        #pdb.set_trace()
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
            #pdb.set_trace()
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
            #pdb.set_trace()
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
        #pdb.set_trace()
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
            #pdb.set_trace()
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

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
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
    
    def forward_rollout(self, x, context_length, rollout_length, mask_type='triangular'):
        src = x[:,:context_length,:]
        # Tensor to hold predictions
        x_hat = torch.zeros(x.shape[0], rollout_length, x.shape[2]).to(x.device)

        for i in range(rollout_length):
            # Use transformer to predict next step, use the last step of context as the tgt
            #tgt = model.transformer(src, src[-1,:,:].unsqueeze(1))
            #out = model.out(tgt)
            out = self.forward(src, src[:,-1,:].unsqueeze(0))
            # Append prediction to predictions tensor
            x_hat[:,i,:] = out
            # Append prediction to context for next prediction
            src = torch.cat((src, out), dim=1)
        
        return x_hat
    
    def loss(self, x, context_length, rollout_length, mask_type):
        src = x[:,:context_length,:]
        t_end = context_length + rollout_length
        if mask_type == 'triangular':
            # shift the tgt by one so we predict the token at pos +1
            tgt = x[:,context_length:(t_end-1),:]
            x_supervise = x[:,(context_length+1):t_end,:]
        elif mask_type == 'square':
            x_supervise = x[:,context_length:t_end,:]
            # feed in a tensor of zeros as the target
            tgt = torch.zeros_like(x_supervise)
        tgt_length = tgt.size(1)
        tgt_mask = self.get_tgt_mask(tgt_length, mask_type='triangular').to(self.device)
        # TEST - add N * num_heads dimensions https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        batch_size = x.size(0)
        tgt_mask = tgt_mask.unsqueeze(0)
        tgt_mask = tgt_mask.expand(batch_size*self.num_heads, -1, -1)
        x_hat = self.forward(src, tgt, tgt_mask)
        # Permute pred to have batch size first again
        x_hat = x_hat.permute(1, 0, 2)   
        
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