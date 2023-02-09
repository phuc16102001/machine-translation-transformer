import torch.nn as nn
import torch

class LabelSmoothingLoss(nn.Module):
    
    def __init__(self, n_class, trg_pad_idx, smoothing = 0.1, dim = -1):
        super().__init__()
        self.n_class = n_class
        self.trg_pad_idx = trg_pad_idx
        self.smoothing = smoothing
        self.confidence = 1 - smoothing
        self.dim = dim
        self.negative_value = smoothing/(self.n_class-2) # Except '<sos>' and '<eos>'
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim = self.dim)
        
        # Construct true_dist
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.negative_value)
            true_dist.scatter_(
                1,
                target.data.unsqueeze(1),
                self.confidence
            )
            
            # Mask '<pad>' of target
            mask = torch.nonzero(
                target.data == self.trg_pad_idx,
                as_tuple = False
            )
            if (mask.dim()>0):
                true_dist.index_fill_(
                    0,
                    mask.squeeze(),
                    0.0
                )
                
        # Negative because log(x) where (x<1) after softmax 
        return torch.mean(torch.sum(-pred * true_dist, dim = self.dim))