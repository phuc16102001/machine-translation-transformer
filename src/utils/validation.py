import torch
from utils.mask import create_mask
import numpy as np

def validiate(model, valid_iter, criterion, src_pad, trg_pad, device):
    model.eval()
    
    with torch.no_grad():
        total_loss = []
        for batch in valid_iter:
            src = batch.src.transpose(0,1).cuda()
            trg = batch.trg.transpose(0,1).cuda()
            trg_input = trg[:, :-1]
            
            src_mask = create_mask(src, src_pad, False, device)
            trg_mask = create_mask(trg_input, trg_pad, True, device)
            preds = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(preds.view(-1, preds.size(-1)), ys)
            loss = loss.item()
            total_loss.append(loss)
    return np.mean(total_loss)