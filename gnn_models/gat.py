import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from gnn_models.modules import mlp

class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.env = args.env
        self.num_humans = args.num_humans
        self.obs_frames = args.obs_frames
        self.human_state_dim = args.feat_dim
        self.encoder = args.encoder

        # architecture settings 
        self.hidden_dim = args.hidden_dim
        if self.encoder == 'mlp':
            self.wh_dims = [4*args.hidden_dim, 2*args.hidden_dim, args.hidden_dim]
            self.w_h = mlp(self.obs_frames*self.human_state_dim, self.wh_dims, last_relu=True)
        elif self.encoder == 'rnn':
            self.w_h = nn.GRU(self.human_state_dim, self.hidden_dim , num_layers=1, batch_first=True)

        if args.gt:
            self.final_layer = mlp(2*self.hidden_dim, [self.hidden_dim, self.hidden_dim//2, self.human_state_dim])
            self.final_layer = torch.nn.Linear(2*self.hidden_dim, self.human_state_dim)
        else:
            self.final_layer = mlp(self.hidden_dim, [self.hidden_dim, self.hidden_dim//2, self.human_state_dim])
            #torch.nn.Linear(self.hidden_dim, human_state_dim)

        self.W = Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # for visualization

    def get_embeddings(self, human_states):
        return self.embedding(human_states)

    def embedding(self, human_states):
        # formerly called encoder
        batch_size = human_states.shape[0]
        num_ts = human_states.shape[1]
        num_humans = human_states.shape[2]
        num_feats = human_states.shape[3]
        # put number of entities in 2nd dim to compute encodings indepenently for every entity
        human_states = human_states.permute(0, 2, 1, 3)
        if self.encoder == 'mlp':
            # flatten time and feature dimensions
            human_states = human_states.reshape(batch_size, num_humans, -1)
            # get an embedding for every entity with an mlp encoding
            embeddings = self.w_h(human_states)
        elif self.encoder == 'rnn':
            # flatten batch and entity dimensions to process entities indepenently 
            human_states = human_states.reshape(batch_size* num_humans, num_ts, num_feats)
            # get an embedding for every entity with an rnn encoding
            embeddings = self.w_h(human_states)
            # get rnn output and take the last timepoint of rnn processing
            embeddings = embeddings[0][:,-1,:]
            # reshape to separate batch and entity dimensions again
            embeddings = embeddings.reshape(batch_size, num_humans, self.hidden_dim)
        return embeddings

    def dynamical(self, embeddings):
        # dynamical model
        X = torch.matmul(torch.matmul(embeddings, self.W), embeddings.permute(0, 2, 1))
        normalized_A = softmax(X, dim=-1)
        next_H = relu(torch.matmul(normalized_A, embeddings))
        return normalized_A, next_H

    def forward(self, human_states, batch_graph):
        # encoder
        embeddings = self.embedding(human_states)
        # dynamical
        normalized_A, next_H = self.dynamical(embeddings)

        # decoder
        if batch_graph is None:
            prev_state = human_states[:, -1, ...]
            return normalized_A, prev_state + self.final_layer(next_H)
        else:
            H = relu(torch.matmul(batch_graph, embeddings))
            prev_state = human_states[:, -1, ...]
            return normalized_A, prev_state + self.final_layer(torch.cat((H, next_H), dim=-1))

    def multistep_forward(self, batch_data, batch_graph, rollouts):
        
        ret = []
        for step in range(rollouts):
            tmp_graph, pred_obs = self.forward(batch_data, batch_graph)
            if step < self.args.edge_types:
                pred_graph = torch.zeros((tmp_graph.shape[0], self.num_humans, self.num_humans-1))
                for i in range(self.num_humans):
                    pred_graph[:, i, 0:i] = tmp_graph[:, i, 0:i].detach()
                    pred_graph[:, i, i:self.num_humans] = tmp_graph[:, i, i+1:self.num_humans].detach()
            
            ret.append([[pred_graph], pred_obs])
            if self.encoder == 'mlp':
                # fixed length encoding - take last T-1 from batch_data and append new pred_obs
                batch_data = torch.cat([batch_data[:, 1:, ...], pred_obs.unsqueeze(1)], dim=1)
            elif self.encoder == 'rnn':
                # fixed length encoding - take last T-1 from batch_data and append new pred_obs
                batch_data = torch.cat([batch_data[:, 1:, ...], pred_obs.unsqueeze(1)], dim=1)
                # # variable length encoding - take all batch_data and append new pred_obs
                # batch_data = torch.cat([batch_data, pred_obs.unsqueeze(1)], dim=1)
        return ret
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        batch_x = x.reshape(-1, x.shape[1], 5, 7)
        batch_context = batch_x[:,:burn_in_length,:,:]
        batch_graph = None
        
        preds = self.multistep_forward(batch_context, batch_graph, rollout_length)

        x_hat = torch.zeros(x.shape[0], rollout_length, x.shape[2])
        for idx in range(rollout_length):
            pred = preds[idx][-1]
            x_hat[:,idx,:] = pred.reshape(x.shape[0], 1, x.shape[2])
            
        return x_hat

    def loss(self, batch_x, burn_in_length, rollout_length):
        loss_fn = torch.nn.MSELoss()
        
        batch_context = batch_x[:,:burn_in_length,:,:]
        true_rollout_x = batch_x[:,-rollout_length:,:,:]
        batch_graph = None
        
        preds = self.multistep_forward(batch_context, batch_graph, rollout_length)
        
        losses = 0.
        for idx in range(rollout_length):
            pred = preds[idx][-1]
            label = true_rollout_x[:, idx, :self.num_humans, :self.human_state_dim]
            loss = loss_fn(pred, label)
            losses += loss
            
        return losses