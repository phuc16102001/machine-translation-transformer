import torch.nn as nn
from modules.norm import Norm
from modules.multihead_attention import MultiHeadAttention
from modules.feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, dropout=dropout)
        
    def forward(self, x, e_out, src_mask, trg_mask):
        x_norm = self.norm_1(x)
        x_attn = self.attn_1(x_norm, x_norm, x_norm, trg_mask)
        x = x + self.dropout_1(x_attn)
        
        x_norm = self.norm_2(x)
        x_attn = self.attn_2(x_norm, e_out, e_out, src_mask)
        x = x + self.dropout_2(x_attn)
        
        x_norm = self.norm_3(x)
        x_ff = self.ff(x_norm)
        x = x + self.dropout_3(x_ff)
        return x