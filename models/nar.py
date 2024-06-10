import torch
import numpy as np
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import random
import torch.nn.functional as F
import math

class new_gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class NAR(nn.Module):
    def __init__(self, hparams, device):
        super().__init__()
        self.hparams = hparams
        self.device = device

        self.token_embeddings = nn.ModuleList([nn.Embedding(self.hparams.nar_vocab_size, embedding_dim=self.hparams.nar_n_emb, padding_idx=0)] * self.hparams.acoustic_num_layer)
        self.wpe = nn.Embedding(2048, self.hparams.nar_n_emb) 
        self.dense_layer = nn.Linear(self.hparams.nar_n_emb * self.hparams.acoustic_num_layer, self.hparams.nar_n_emb, bias=False)

        self.res_embedding = nn.Embedding(self.hparams.acoustic_num_layer - 1, self.hparams.nar_n_emb)

        activation = new_gelu()
        self.transformer_layers = nn.ModuleList()
        for i in range(self.hparams.nar_n_layer):
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    d_model=self.hparams.nar_n_emb,
                    nhead=self.hparams.nar_n_head ,
                    dim_feedforward=self.hparams.nar_n_emb * 4,
                    dropout=0.1,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                )
            )
        self.mlp_layer = nn.Linear(self.hparams.nar_n_emb, self.hparams.num_acoustic + 2, bias=False)  # quant_token_num + eos


    def inference(self, data):
        # unpack data
        init_full_idx = data['full_idx']
        pos_ids = data['pos_ids']
        tags = data['tags']
        prompt_len = data['prompt_len']
        
        # Calculate lengths
        semantic_len = (tags == 1).sum(dim=1)
        acoustic_len = (tags == 2).sum(dim=1)
        
        # Initialize the sequence tensor with indices for each acoustic layer
        full_idx = torch.stack([data['first_idx']] * self.hparams.acoustic_num_layer, dim=1)
        full_idx[:, :, semantic_len[0]:semantic_len[0] + prompt_len[0]] = init_full_idx[:,semantic_len[0]:, :].transpose(1, 2)
        
        batch_size, layer_size, t_size = full_idx.size()
        
        # Create masks
        layer_index = torch.ones(size=[batch_size,], device=self.device)
        layer_mask = layer_index.unsqueeze(1) > torch.arange(self.hparams.acoustic_num_layer, device=self.device).unsqueeze(0)
        prompt_mask = (semantic_len + prompt_len).unsqueeze(1) > torch.arange(t_size, device=self.device).unsqueeze(0)
        
        # Combine masks
        mask = layer_mask.unsqueeze(2) + prompt_mask.unsqueeze(1)
        full_idx = torch.where(mask, full_idx, torch.zeros_like(full_idx))
        
        for layer_index in range(1, self.hparams.acoustic_num_layer):
            layer_index_tensor = torch.LongTensor([layer_index, ]).repeat(batch_size, ).to(self.device)
            layer_mask = layer_index_tensor.unsqueeze(1) > torch.arange(self.hparams.acoustic_num_layer, device=self.device).unsqueeze(0)
            
            # Apply masks
            mask = layer_mask.unsqueeze(2) + prompt_mask.unsqueeze(1)
            mask_full_idx = torch.where(mask, full_idx, torch.zeros_like(full_idx))
            
            # Generate embeddings
            embeddings = [self.token_embeddings[i](mask_full_idx[:, i, :]) for i in range(layer_size)]
            embeddings = torch.cat(embeddings, dim=-1)
            embeddings = self.dense_layer(embeddings)
            
            # Add layer and position embeddings
            res_embeddings = self.res_embedding(layer_index_tensor - 1).unsqueeze(1)
            outputs = embeddings + self.wpe(pos_ids) + res_embeddings
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                outputs = layer(outputs)
            
            logits = self.mlp_layer(outputs)
            logits[:, :, -2:] = -float('Inf') # unused token
            
            # NAR greedy search
            samples = torch.argmax(logits, dim=-1) + self.hparams.num_semantic + 1
            samples = torch.where(prompt_mask, full_idx[:, layer_index, :], samples)
            full_idx[:, layer_index, :] = samples
        
        return full_idx[:, :, semantic_len[0]:]
