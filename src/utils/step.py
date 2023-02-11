from utils.mask import create_mask

def step(model, opt, batch, criterion, src_pad, trg_pad, device):
    model.train()
    
    src = batch.src.transpose(0, 1) # convert to (bs x seq_len)
    trg = batch.trg.transpose(0, 1) # convert to (bs x seq_len)
    trg_input = trg[:, :-1] # Exclude last word to predict
    
    # Create mask
    src_mask = create_mask(src, src_pad, False, device)
    trg_mask = create_mask(trg_input, trg_pad, True, device)
    
    # Predict
    preds = model(src, trg_input, src_mask, trg_mask)
    preds = preds.view(-1, preds.size(-1))
    
    # Calculate loss
    ys = trg[:, 1:].contiguous().view(-1) # Exclude the first word
    opt.zero_grad()
    loss = criterion(preds, ys)
    loss.backward()
    opt.step_and_update_lr()
    
    loss = loss.item()
    return loss