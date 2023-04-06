# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from models.caption.containers import Module
from einops import rearrange, repeat


def init_params(module):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'm_' in name:  # for memory
            nn.init.normal_(param, mean=0, std=0.01)


class Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, n_heads, dropout=0.2, **kwargs):
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None, attention_weights=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n)
        q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
        k = rearrange(self.fc_k(k), 'b nk (head d) -> b head nk d', head=self.n_heads)
        v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)  # [b h nq nk]
        if attention_weights is not None:
            scores = scores * attention_weights
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.bool(), -np.inf)
        p_attn = torch.softmax(scores, -1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')

        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MemoryAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_memories, dropout=0.0, **kwargs):
        # * adapted from Meshed-Memory Transformers; n_memories: # mem slots
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        if n_memories > 0:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_memories = n_memories
        self.d_k = d_model // n_heads

        self.apply(init_params)

    def forward(self, q, k, v, attention_mask=None, attention_weights=None):
        # q, k, v: (b, n, d_model), mask: (b, n, n) - True indicates masking

        b_s, nq = q.shape[:2]
        nk = k.shape[1]
        if self.n_memories > 0:
            m_k = repeat(self.m_k, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.d_k)
            m_v = repeat(self.m_v, '() m d_model -> b m d_model', b=q.shape[0]) * np.sqrt(self.n_memories)
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)

            k = torch.cat([self.fc_k(k), m_k], 1)
            v = torch.cat([self.fc_v(v), m_v], 1)
            k = rearrange(k, 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(v, 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
            if attention_weights is not None:
                scores = torch.cat([scores[:, :, :, :nk] * attention_weights, scores[:, :, :, nk:]], dim=-1)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -np.inf)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            scores = torch.matmul(q, k) / np.sqrt(self.d_k)  # [b h nq nk]
            if attention_weights is not None:
                scores = scores * attention_weights
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -np.inf)

        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # [b h nq nk] * [b h nk dv] = [b h nq dv] -> [b nq h dv] -> [b nq h*dv]
        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(Module):

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=.1,
        can_be_stateful=False,  # for fast inference
        attention_module=Attention,  # Attention or MemoryAttention
        attn_dropout=0.0,
        **kwargs,
    ):
        super(MultiHeadAttention, self).__init__()
        attention_module = attention_module or Attention
        self.attention = attention_module(d_model=d_model, n_heads=n_heads, dropout=attn_dropout, **kwargs)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:  # store prev computed K & V for fast inference
            self.register_state('running_keys', torch.zeros((1, d_model)))
            self.register_state('running_values', torch.zeros((1, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            # keys, values:             from the current input token: [B, 1, D]
            # running_keys, values:     from prev tokens: [B, t-1, D]
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            self.running_values = torch.cat([self.running_values, values], 1)
            if self.timestep == 0:
                keys = self.running_keys = self.running_keys[:, 1:]  # [B t D]
                values = self.running_values = self.running_values[:, 1:]  # [B t D]
            else:
                keys = self.running_keys  # [B t D]
                values = self.running_values  # [B t D]

            self.timestep += 1

        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out