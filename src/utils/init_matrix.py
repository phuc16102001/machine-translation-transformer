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
    out = out[:, -1] # (bs x 1 x trg_vocab) --> Softmax
    
    # Get top-K (default topk use last dimension)
    # Because out is (bs x 1 x trg_vocab) then
    # probs and idx also (bs x k)
    # That is why we must get [0]
    probs, idx = out.data.topk(k) # (Value tensor, Index tensor) 
    log_scores = []
    for p in probs.data[0]:
        log_scores.append(math.log(p))
    log_scores = torch.Tensor(log_scores)
    log_scores = log_scores.unsqueeze(1)    # (k x 1)
    
    # Construct K outputs
    k_res = torch.zeros(k, max_len).long()    # (k x max_trg_len)
    k_res = k_res.to(device)
    k_res[:, 0] = init_token  # First words are <sos>
    k_res[:, 1] = idx[0]      # Second words are top-K

    # Construct e_out to k-size batch
    e_out_bs = torch.zeros(k, e_out.size(-2), e_out.size(-1))
    e_out_bs = e_out_bs.to(device)
    e_out_bs[:, :] = e_out
    
    return e_out_bs, k_res, log_scores