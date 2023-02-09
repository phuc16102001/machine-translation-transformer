import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        
    def forward(self, x):
        num = (x - x.mean(dim = -1, keepdim = True))
        denom = (x.std(dim=-1, keepdim=True) + self.eps) 
        out = self.alpha * (num / denom) + self.bias
        return out