import torch

from modules.transformer import Transformer
from config.config import config
from utils.tokenizer import tokenizer
from utils.field import create_field

def main():
    vi_tokenizer = tokenizer('vi_core_news_lg')
    en_tokenizer = tokenizer('en_core_web_sm')
    src_field, trg_field = create_field(vi_tokenizer, en_tokenizer)

    src_vocab = torch.load('../models/src_vocab.pth')
    trg_vocab = torch.load('../models/trg_vocab.pth')
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    d_model = config['d_model']

    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = d_model,
        n = 6,
        heads = 8,
        dropout = 0.1
    )
    model_ckpt = torch.load('../models/model_best.pt')
    print(model_ckpt)
    model.load_state_dict(model_ckpt)

if __name__=="__main__":
    main()