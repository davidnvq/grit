import torch
from torch import nn
from torch.nn import functional as F
from models.common.attention import MultiHeadAttention
from models.common.pos_embed import PositionWiseFeedForward
from einops import repeat


class TransformerLayer(nn.Module):

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        attention_module=None,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()
        self.mhatt = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attention_module=attention_module,
            attn_dropout=attn_dropout,
            **kwargs,
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class GridFeatureNetwork(nn.Module):

    def __init__(
        self,
        n_layers,
        pad_idx,
        d_in=1024,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        attn_dropout=0.0,
        attention_module=None,
        **kwargs,
    ):
        super().__init__()
        self.fc = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                attention_module=attention_module,
                **kwargs,
            ) for _ in range(n_layers)
        ])

    def forward(self, input, attention_mask=None, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)

        if attention_mask is None:
            attention_mask = (torch.sum(out, dim=-1) == self.padding_idx)
            attention_mask = repeat(attention_mask, 'B N -> B 1 1 N')  # [B Head Nq N]

        outs = []
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask