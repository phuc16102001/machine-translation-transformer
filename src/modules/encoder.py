import torch.nn as nn
from utils.cloner import get_clones
from modules.embedder import Embedder
from modules.positional_encoding import PositionalEncoder
from modules.encoder_layer import EncoderLayer
from modules.norm import Norm

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n, heads, dropout=0.1):
        super().__init__()
        
        self.n = n
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.encoder_layers = get_clones(
            EncoderLayer(d_model, heads, dropout),
            n
        )
        self.norm = Norm(d_model)
    
    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pe(x)
        for i in range(self.n):
            x = self.encoder_layers[i](x, mask)
        x = self.norm(x)
        return x