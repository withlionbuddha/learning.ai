# decoder_layer.py

import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        enc_attn_output = self.encoder_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_attn_output))
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
