import torch
import torch.nn as nn

from utils.reader import create_data
from utils.tokenizer import tokenizer
from utils.field import create_field
from utils.dataset import create_dataset
from utils.scheduler import MyScheduler
from utils.loss import LabelSmoothingLoss
from utils.step import step
from utils.validation import validiate

from modules.transformer import Transformer
import shutil
import json
import os
from argparse import ArgumentParser

def main(args):
    data_folder = args.data_folder
    model_folder = args.model_folder

    print("Load config")
    config_file = open('config/config.json')
    cfg = json.load(config_file)
    d_model = cfg['d_model']
    batch_size = cfg['batch_size']
    max_strlen = cfg['max_strlen']
    n_epoch = cfg['epoch']
    print_every = cfg['print_every']
    n_layers = cfg['n_layers']
    heads = cfg['heads']
    dropout = cfg['dropout']
    print(json.dumps(cfg, indent=3))

    device = 'cpu'
    if (torch.cuda.is_available()): 
        device = 'cuda'
    print(f"Use {device}")

    print("Loading data")
    train_src = os.path.join(data_folder, 'train.vi')
    train_trg = os.path.join(data_folder, 'train.en')
    val_src = os.path.join(data_folder, 'tst2013.vi')
    val_trg = os.path.join(data_folder, 'tst2013.en')
    df_train = create_data(train_src, train_trg)
    df_val = create_data(val_src, val_trg)
    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")

    print("Creating tokenizer")
    vi_tokenizer = tokenizer('vi_core_news_lg')
    en_tokenizer = tokenizer('en_core_web_sm')
    
    print("Creating field")
    src_field, trg_field = create_field(vi_tokenizer, en_tokenizer)

    print("Construct dataset")
    train_dataset = create_dataset(df_train, src_field, trg_field, batch_size, max_strlen, device)
    val_dataset = create_dataset(df_val, src_field, trg_field, batch_size, max_strlen, device, istrain=False)
    src_pad = src_field.vocab.stoi['<pad>']
    trg_pad = trg_field.vocab.stoi['<pad>']
    print(f"Source vocabulary size: {len(src_field.vocab)}")
    print(f"Target vocabulary size: {len(trg_field.vocab)}")
    torch.save(src_field.vocab, os.path.join(model_folder, 'src_vocab.pth'))
    torch.save(trg_field.vocab, os.path.join(model_folder, 'trg_vocab.pth'))

    print(f"Creating model with d_model = {d_model}")
    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = d_model,
        n = n_layers,
        heads = heads,
        dropout = dropout
    )
    model = model.to(device)
    shutil.copyfile('config/config.json', os.path.join(model_folder, 'config.json'))

    print("Init parameters")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print("Creating optimizer")
    opt = MyScheduler(
        torch.optim.Adam(
            model.parameters(), 
            betas = (0.9, 0.98), 
            eps = 1e-9
        ),
        init_lr=0.2, 
        d_model=d_model, 
        n_warmup=4000
    )

    print("Creating loss function")
    criterion = LabelSmoothingLoss(
        len(trg_field.vocab), 
        trg_pad_idx = trg_pad, 
        smoothing = 0.1
    )

    best_loss = None
    print("Training begin")
    for epoch in range(n_epoch):
        total_loss = 0
        
        n_iter = 0
        for i, batch in enumerate(train_dataset): 
            loss = step(model, opt, batch, criterion, src_pad, trg_pad, device)
            n_iter += 1
            total_loss += loss
            if ((i+1)%print_every==0):
                avg_loss = total_loss / n_iter
                total_loss = 0
                n_iter = 0
                print("epoch: {:03d} - iter: {:04d} - loss: {:.4f}".format(epoch+1, i+1, avg_loss))

        valid_loss = validiate(model, val_dataset, criterion, src_pad, trg_pad, device)
        print('epoch: {:03d} - valid loss: {:.4f}'.format(epoch+1, valid_loss))
        torch.save(model.state_dict(), os.path.join(model_folder, f'model.{epoch+1}.pt'))
        if ((best_loss is None) or (best_loss>valid_loss)):
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(model_folder, f'model_best.pt'))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_folder',
        type=str,
        required=True,
        help="Path to data folder"
    )
    parser.add_argument(
        '-m',
        '--model_folder',
        type=str,
        required=True,
        help="Path to model folder"
    )
    args = parser.parse_args()
    main(args)