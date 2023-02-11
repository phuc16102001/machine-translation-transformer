import torch
import torch.nn as nn

from modules.transformer import Transformer
from utils.tokenizer import tokenizer
from utils.field import create_field
from utils.translator import translate

from argparse import ArgumentParser
import json

def main(args):
    config_file = open('../models/config.json')
    cfg = json.load(config_file)
    max_strlen = cfg['max_strlen']
    k = cfg['k']
    device = cfg['device']

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
    device = cfg['device']
    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = cfg['d_model'],
        n = cfg['n_layers'],
        heads = cfg['heads'],
        dropout = cfg['dropout']
    )
    model_ckpt = torch.load('../models/model_best.pt')
    model.load_state_dict(model_ckpt)
    
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.to(device)    

    print("Running inference")
    sentence = args.prompt
    print(f"Input: {sentence}")
    print(f"Output: {translate(sentence, model, src_field, trg_field, max_strlen, device, k)}")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        required=True,
        help="Prompt text to translate (VN)"
    )
    args = parser.parse_args()
    main(args)