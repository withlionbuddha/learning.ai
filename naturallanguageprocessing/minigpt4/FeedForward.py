# feed_forward.py
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
