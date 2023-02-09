import math
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None, dropout=None):
    """
    q: batch x head x seq_len x d_model
    k: batch x head x seq_len x d_model
    v: batch x head x seq_len x d_model

    mask: batch x 1 x seq_len
    output: batch x head x seq_len x d_model
    """
    d_k = q.size(-1) # last dimension
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k) # batch x head x seq_len x seq_len
    
    # Mask the future scores (to train model) and padding
    if (mask is not None):
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Softmax will convert all -inf to 0
    scores = F.softmax(scores, dim = -1) # Softmax over last dimension
    
    if (dropout is not None):
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores