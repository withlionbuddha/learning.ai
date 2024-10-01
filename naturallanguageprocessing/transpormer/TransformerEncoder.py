
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from EncoderLayer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
