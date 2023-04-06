import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from models.common.attention import MultiHeadAttention, Attention
from models.common.pos_embed import sinusoid_encoding_table, FeedForward
from models.caption.containers import Module, ModuleList


class GeneratorLayer(Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__()

        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_ff, dropout)


class ParallelAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, activation='sigmoid', n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        self_att = self.self_att(x, x, x, mask_x)
        self_att = self_att * mask_pad

        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad

        # [B, N, D]
        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att2], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2) / np.sqrt(2)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class ConcatAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)
        self.vis_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)

    def forward(self, x, y, mask_pad, mask_x, mask_y):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att(out, y, y, mask_y) * mask_pad
        out = self.pwff(out) * mask_pad
        return out


class SequentialAttentionLayer(GeneratorLayer):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1, n_memories=0):
        super().__init__(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_memories=0)

        self.vis_att1 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.vis_att2 = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att1(out, y1, y1, mask_y1) * mask_pad
        out = self.vis_att2(out, y2, y2, mask_y2) * mask_pad
        ff = self.pwff(out)
        ff = ff * mask_pad
        return ff


class CaptionGenerator(Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'parallel': ParallelAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }

    def __init__(self,
                 vocab_size,
                 max_len,
                 n_layers,
                 pad_idx,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 dropout=.1,
                 decoder_name='parallel',
                 cfg=None):
        super().__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.cfg = cfg
        self.decoder_name = decoder_name
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]

        self.layers = ModuleList([generator_layer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers

        self.register_state('running_mask_x', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def get_seq_inputs(self, input):
        # input (b_s, seq_len); when use beam search: input [BB 1]
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_x = mask_x.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        x = self.word_emb(input) + self.pos_emb(seq)

        return x, mask_x, mask_pad

    def forward(self, input, vis_inputs):
        x, mask_x, mask_pad = self.get_seq_inputs(input)

        if self.decoder_name == 'concat':
            y = torch.cat([vis_inputs['grid_feat'], vis_inputs['reg_feat']], dim=1)
            mask_y = torch.cat([vis_inputs['gri_mask'], vis_inputs['reg_mask']], dim=3)

            for layer in self.layers:
                x = layer(x, y, mask_pad, mask_x, mask_y)

        if self.decoder_name == 'sequential':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)

        if self.decoder_name == 'parallel':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']

            for layer in self.layers:
                x = layer(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
