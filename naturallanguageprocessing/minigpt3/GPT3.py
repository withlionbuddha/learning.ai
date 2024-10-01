import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock

class GPT3(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, ff_hidden_size, max_length, dropout):
        super(GPT3, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, ff_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        logits = self.fc_out(out)
        return logits
