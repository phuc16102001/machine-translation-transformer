from utils.mask import create_mask, create_np_mask
import torch
import torch.nn.functional as F
import math

def create_init_matrix(src, model, src_field, trg_field, device, k, max_len):
    """
    Use to create the first level of top-K beam search (for Bleu score) 
    This is also because you need to calculate e_output once only
    """
    init_token = trg_field.vocab.stoi['<sos>'] # Start of sentence token
    src_pad = src_field.vocab.stoi['<pad>']
    src_mask = create_mask(src, src_pad, False, device)
    trg_mask = create_np_mask(1, device) # First word mask
    
    # Predict first output
    trg = torch.LongTensor([[init_token]])
    trg = trg.to(device)
    e_out = model.encoder(src, src_mask)
    d_out = model.decoder(trg, e_out, src_mask, trg_mask)
    out = model.out(d_out) # bs x seq_len x trg_vocab_size
    out = F.softmax(out, dim = -1)
    
    # Extract last word
    out = out[:, -1]
    
    # Get top-K
    probs, idx = out.data.topk(k) # (Value tensor, Index tensor)
    log_scores = []
    for p in probs.data[0]:
        log_scores.append(math.log(p))
    log_scores = torch.Tensor(log_scores)
    log_scores = log_scores.unsqueeze(0)
    
    # Construct K outputs
    out = torch.zeros(k, max_len).long()
    out = out.to(device)
    out[:, 0] = init_token # First words are <sos>
    out[:, 1] = idx[0]   # Second words are top-K
    
    return e_out