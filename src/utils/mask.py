import numpy as np
import torch
from torch.autograd import Variable

def create_np_mask(n, device):
    mask = np.triu(
        np.ones(shape=(1, n, n)),
        k=1
    ).astype('uint8')
    mask = Variable(torch.from_numpy(mask)==0)
    mask = mask.to(device)
    return mask

def create_mask(idx_list, idx_pad, is_trg, device):
    mask = (idx_list != idx_pad).unsqueeze(-2)
    mask = mask.to(device)
    if (is_trg):
        size = idx_list.size(1) # seq_len
        np_mask = create_np_mask(size, device)
        if (mask.is_cuda): 
            np_mask.cuda()
        mask = mask & np_mask
    return mask