import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n, heads, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, n, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
        
    def forward(self, src, trg, src_mask, trg_mask):
        e_out = self.encoder(src, src_mask)
        d_out = self.decoder(trg, e_out, src_mask, trg_mask)
        out = self.out(d_out)
        return out