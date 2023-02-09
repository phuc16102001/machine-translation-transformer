import torch
from modules.transformer import Transformer
from config.config import config

def main():
    src_field = torch.load('../models/src_vocab.pth')
    trg_field = torch.load('../models/trg_vocab.pth')
    d_model = config['d_model']

    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = d_model,
        n = 6,
        heads = 8,
        dropout = 0.1
    )
    model.load_state_dict(torch.load('../models/model_best.pth'))

if __name__=="__main__":
    main()