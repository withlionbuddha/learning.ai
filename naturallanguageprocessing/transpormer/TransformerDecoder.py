# transformer_decoder.py

import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from DecoderLayer import DecoderLayer

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.fc_out(x)
