import torch.nn as nn
from utils.attention import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        assert(d_model % heads == 0)
        
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model//heads # In order to d_k.heads = d_model
        
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask):
        """
        q: bs x seq_len x d_model
        k: bs x seq_len x d_model
        v: bs x seq_len x d_model
        """
        
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.heads, self.d_k)
        k = self.w_k(k).view(bs, -1, self.heads, self.d_k)
        v = self.w_v(v).view(bs, -1, self.heads, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)        
        v = v.transpose(1, 2)

        # Attention mechanism
        output_attn, self.scores = attention(q, k, v, mask, self.dropout)
        
        concatenate_tensor = output_attn.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concatenate_tensor)
        return output