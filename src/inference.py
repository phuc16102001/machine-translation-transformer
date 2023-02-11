import torch
import torch.nn as nn

from modules.transformer import Transformer
from utils.tokenizer import tokenizer
from utils.field import create_field
from utils.translator import translate

from argparse import ArgumentParser
import json
import os

def main(args):
    model_folder = args.model_folder
    sentence = args.prompt

    config_file = open(os.path.join(model_folder, 'config.json'))
    cfg = json.load(config_file)
    max_strlen = cfg['max_strlen']
    k = cfg['k']

    device = 'cpu'
    if (torch.cuda.is_available()): 
        device = 'cuda'

    print("Creating tokenizer")
    vi_tokenizer = tokenizer('vi_core_news_lg')
    en_tokenizer = tokenizer('en_core_web_sm')
    src_field, trg_field = create_field(vi_tokenizer, en_tokenizer)

    print("Loading vocabulary")
    src_vocab = torch.load(os.path.join(model_folder, 'src_vocab.pth'))
    trg_vocab = torch.load(os.path.join(model_folder, 'trg_vocab.pth'))
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab

    print("Loading model")
    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = cfg['d_model'],
        n = cfg['n_layers'],
        heads = cfg['heads'],
        dropout = cfg['dropout']
    )

    model_ckpt = torch.load(os.path.join(model_folder, 'model_best.pt'))
    model.load_state_dict(model_ckpt)
    model = model.to(device)    

    print("Running inference")
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
    parser.add_argument(
        '-m',
        '--model_folder',
        type=str,
        required=True,
        help='Path to the model folder'
    )
    args = parser.parse_args()
    main(args)