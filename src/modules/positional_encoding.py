import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length = 200, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0) # Expand batch size dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1) # (batch, len, d_model)
        pe = Variable(
            self.pe[:, :seq_len],
            requires_grad = False
        )
        if (x.is_cuda): pe.cuda()
        x = x * math.sqrt(self.d_model)
        x = x + pe
        x = self.dropout(x)
        return x