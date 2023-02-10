import torch

from modules.transformer import Transformer
from config.config import config
from utils.tokenizer import tokenizer
from utils.field import create_field
from utils.translator import translate

def main():
    print("Creating tokenizer")
    vi_tokenizer = tokenizer('vi_core_news_lg')
    en_tokenizer = tokenizer('en_core_web_sm')
    src_field, trg_field = create_field(vi_tokenizer, en_tokenizer)

    print("Loading vocabulary")
    src_vocab = torch.load('../models/src_vocab.pth')
    trg_vocab = torch.load('../models/trg_vocab.pth')
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    print("Loading model")
    d_model = config['d_model']
    device = config['device']
    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = d_model,
        n = 6,
        heads = 8,
        dropout = 0.1
    )
    model_ckpt = torch.load('../models/model_best.pt')
    model.load_state_dict(model_ckpt)
    model = model.to(device)

    print("Running inference")
    sentence = "Xin chào bạn, tui là Đỗ Vương Phúc. Tớ là sinh viên năm cuối. Hiện đang học chuyên ngành Khoa học máy tính"
    print(f"Input: {sentence}")
    print(f"Output: {translate(sentence, model, src_field, trg_field)}")

if __name__=="__main__":
    main()