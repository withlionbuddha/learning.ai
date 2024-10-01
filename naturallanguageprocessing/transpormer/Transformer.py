import torch
import torch.nn as nn
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder


# 전체 Transformer 구현
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output