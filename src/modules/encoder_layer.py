from modules.multihead_attention import MultiHeadAttention
from modules.norm import Norm
from modules.feed_forward import FeedForward

import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.norm_1 = Norm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm_2 = Norm(d_model)
    
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_norm = self.norm_1(x)
        x_attn = self.attn(x_norm, x_norm, x_norm, mask)
        
        x = x + self.dropout_1(x_attn)
        x_norm = self.norm_2(x)
        x_ff = self.ff(x_norm)

        x = x + self.dropout_2(x_ff)
        return x