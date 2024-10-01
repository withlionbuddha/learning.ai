import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(N, -1, self.d_model)

        return self.fc_out(out)
