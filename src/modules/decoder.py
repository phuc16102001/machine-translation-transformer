import torch.nn as nn
from modules.embedder import Embedder
from modules.positional_encoding import PositionalEncoder
from modules.decoder_layer import DecoderLayer
from modules.norm import Norm
from utils.cloner import get_clones

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads, dropout=0.1):
        super().__init__()
        
        self.n = n
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.decoder_layers = get_clones(
            DecoderLayer(d_model, heads, dropout),
            n
        )
        self.norm = Norm(d_model)
    
    def forward(self, x, e_out, src_mask, trg_mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.n):
            x = self.decoder_layers[i](x, e_out, src_mask, trg_mask)
        x = self.norm(x)
        return x