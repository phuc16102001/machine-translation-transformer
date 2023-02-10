from utils.init_matrix import create_init_matrix
from utils.mask import create_mask, create_np_mask
import torch.nn.functional as F
import torch
import math

def find_best_k(k_res, last_word_out, current_log_scores, current_length, k):
    
    # Result as (k x k) because topk use last dimension only
    probs, last_word_idx = last_word_out.data.topk(k)
    
    # Calculate log probs
    log_scores = []
    for p in probs.data.view(-1):
        log_scores.append(math.log(p))
    log_scores = torch.Tensor(log_scores)
    log_scores = log_scores.view(k, -1)   # Reshape back to (k x k)
    log_scores += current_log_scores     # Calculate total log probs (broadcasting)

    # Find topk total log score
    log_scores = log_scores.view(-1)
    k_probs, k_idx = log_scores.topk(k)
    row = k_idx // k
    col = k_idx % k
    new_log_scores = k_probs.unsqueeze(1)

    # Create new result from list of row, col
    k_res[:, :current_length] = k_res[row, :current_length] # Keep the row in topk for previous words
    k_res[:, current_length] = last_word_idx[row, col]

    return k_res, new_log_scores

def beam_search(sentence, model, src_field, trg_field, device, k, max_strlen):
    
    # Dimension #
    # e_out:        k x seq_len x d_model
    # out:          k x max_len
    # log_scores:   k x 1
    #############
    e_out, k_res, log_scores = create_init_matrix(sentence, model, src_field, trg_field, device, k, max_strlen)
    
    src_pad = src_field.vocab.stoi['<pad>']
    eos_token = trg_field.vocab.stoi['<eos>']
    src_mask = create_mask(sentence, src_pad, False, device)

    for i in range(2, max_strlen):
        trg_mask = create_np_mask(i, device)
        d_out = model.decoder(k_res[:, :i], e_out, src_mask, trg_mask)
        out = model.out(d_out)
        out = F.softmax(out, dim = -1)

        # Extract last word
        out = out[:, -1] # (bs x 1 x trg_vocab) --> Softmax

        # Find topk result
        k_res, log_scores = find_best_k(k_res, out, log_scores, i, k)

        # Find result length
        end_row_col = (k_res == eos_token).nonzero()
        sentence_lengths = torch.zeros(k, dtype=torch.long).to(device)
        for value in end_row_col:
            sentence_idx = value[0] 
            sentence_len = value[1]
            sentence_lengths[sentence_idx] = sentence_len
        n_finish = len([length for length in sentence_lengths if length>0])     # Number of finished sentences

        # End searching and find the best choice
        best_idx = 0
        if (n_finish == k):
            alpha = 0.7
            denom = (sentence_lengths ** alpha).to(device)
            log_scores = log_scores.view(-1).to(device)
            rate = (log_scores / denom).unsqueeze(0)
            _, best_idx = torch.max(rate, 1)
            break

    best_sentence = k_res[best_idx].view(-1)
    length = (best_sentence==eos_token).nonzero()[0]
    return best_sentence, length
