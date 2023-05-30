from torch.autograd import Variable
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import softmax, relu
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import itertools
import logging
import math
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from gnn_models.modules import *


class RFM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_humans = args.num_humans
        self.obs_frames = args.obs_frames

        self.edge_types = args.edge_types
        self.skip_first = args.skip_first
        self.hidden_dim = args.hidden_dim
        self.human_state_dim = args.feat_dim
        self.input_human_state_dim = args.feat_dim

        self.timesteps = args.obs_frames
        self.dims = args.hidden_dim
        self.encoder_hidden = args.hidden_dim
        self.decoder_hidden = args.hidden_dim
        self.encoder_dropout = 0.
        self.decoder_dropout = 0.
        self.factor = True

        if args.encoder == 'cnn':
            self.encoder_type = 'cnn'
            self.encoder = CNNEncoder(self.input_human_state_dim,
                                      self.encoder_hidden,
                                      self.edge_types,
                                      self.encoder_dropout,
                                      self.factor)
        elif args.encoder == 'rnn':
            self.encoder_type = 'rnn'
            self.encoder = RNNEncoder(self.input_human_state_dim,
                                      self.encoder_hidden,
                                      self.edge_types,
                                      self.encoder_dropout,
                                      self.factor)
        else:
            self.encoder_type = 'mlp'
            self.encoder = MLPEncoder(self.obs_frames * self.input_human_state_dim,
                                      self.encoder_hidden,
                                      self.edge_types,
                                      self.encoder_dropout,
                                      self.factor)

        self.rnn_decoder = RNNDecoder(n_in_node=self.decoder_hidden,
                                      edge_types=self.edge_types,
                                      n_hid=self.decoder_hidden,
                                      do_prob=self.decoder_dropout,
                                      skip_first=self.skip_first)

        self.trans = nn.Linear(self.encoder_hidden, self.decoder_hidden)
        self.out_fc3 = nn.Linear(self.decoder_hidden, self.human_state_dim)

        off_diag = np.ones([self.num_humans, self.num_humans]) - np.eye(self.num_humans)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(self.device)
        self.rel_send = torch.FloatTensor(rel_send).to(self.device)
        
    def adjust_sequence_length(self, batch_context):
        batch_size, seq_len, _, _ = batch_context.size()
        target_len = self.obs_frames
    
        if seq_len < target_len:
            # If sequence length is less than 50, pad with zeros at the beginning
            padding = torch.zeros((batch_size, target_len - seq_len, self.num_humans, self.human_state_dim), device=batch_context.device)
            batch_context = torch.cat((padding, batch_context), dim=1)
        elif seq_len > target_len:
            # If sequence length is more than 50, take the last 50 frames
            batch_context = batch_context[:, -target_len:, :, :]
    
        return batch_context

    def encode(self, batch_data):
        batch_data = batch_data.permute(0, 2, 1, 3)
        node_embeddings, logits = self.encoder(batch_data.contiguous(), self.rel_rec, self.rel_send)
        return node_embeddings, logits

    def multistep_forward(self, batch_data, batch_graph, rollouts):
        # batch_size, obs_frmes, num_humans, feat_dim
        assert batch_data.requires_grad is False
        batch_size = batch_data.shape[0]
        batch_data = batch_data.permute(0, 2, 1, 3)

        if self.encoder_type == 'rnn':
            node_embeddings, logits, x_all = self.encoder(batch_data.contiguous(), self.rel_rec, self.rel_send)
        else:
            node_embeddings, logits = self.encoder(batch_data.contiguous(), self.rel_rec, self.rel_send)
        edges = F.softmax(logits, dim=-1)
        num_layers = edges.shape[-1]
        pred_graphs = [edges[..., i].reshape(edges.shape[0], self.num_humans, self.num_humans-1) for i in range(num_layers)]

        if batch_graph is not None:
            edges = torch.zeros(logit[..., 0].shape).to(self.device)
            idx = 0
            for i in range(self.num_humans):
                for j in range(self.num_humans):
                    if i == j: continue
                    edges[:, idx] = batch_graph[:, i, j]
                    idx += 1
            pred_graphs[-1] = batch_graph


        node_embeddings = self.trans(node_embeddings)
        output = self.rnn_decoder(node_embeddings, edges,
                                  self.rel_rec, self.rel_send,
                                  pred_steps=rollouts,
                                  dynamic_graph=False,
                                  encoder=None,
                                  burn_in=False,
                                  burn_in_steps=0)
        ret = []
        for step in range(rollouts):
            pred = self.out_fc3(output[:, :, step, :])
            if step == 0:
                pred = batch_data[:, :, -1, :self.human_state_dim] + pred
            else:
                pred = ret[-1][-1] + pred
            ret.append((pred_graphs, pred))

        return ret
    
    def forward_rollout(self, x, burn_in_length, rollout_length):
        batch_x = x.reshape(-1, x.shape[1], 5, 7)
        batch_context = batch_x[:,:burn_in_length,:,:]
        batch_graph = None
        
        if self.encoder_type == 'mlp' and burn_in_length != self.obs_frames:
            batch_context = self.adjust_sequence_length(batch_context)
        
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
