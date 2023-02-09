import torch
import torch.nn as nn

from config.config import config

from utils.reader import create_data
from utils.tokenizer import tokenizer
from utils.field import create_field
from utils.dataset import create_dataset
from utils.scheduler import MyScheduler
from utils.loss import LabelSmoothingLoss
from utils.step import step
from utils.validation import validiate

from modules.transformer import Transformer

def main():
    device = config['device']
    d_model = config['d_model']
    batch_size = config['batch_size']
    max_strlen = config['max_strlen']
    n_epoch = config['epoch']
    print_every = config['print_every']

    print("Loading data")
    df_train = create_data('../data/train.vi', '../data/train.en')
    df_val = create_data('../data/tst2013.vi', '../data/tst2013.en')
    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")

    print("Creating tokenizer")
    vi_tokenizer = tokenizer('vi_core_news_lg')
    en_tokenizer = tokenizer('en_core_web_sm')
    
    print("Creating field")
    src_field, trg_field = create_field(vi_tokenizer, en_tokenizer)

    print("Construct dataset")
    train_dataset = create_dataset(df_train, src_field, trg_field)
    val_dataset = create_dataset(df_val, src_field, trg_field, istrain=False)
    src_pad = src_field.vocab.stoi['<pad>']
    trg_pad = trg_field.vocab.stoi['<pad>']
    print(f"Source vocabulary size: {len(src_field.vocab)}")
    print(f"Target vocabulary size: {len(trg_field.vocab)}")
    torch.save(src_field.vocab, '../models/src_vocab.pth')
    torch.save(trg_field.vocab, '../models/trg_vocab.pth')

    print(f"Creating model with d_model = {d_model}")
    model = Transformer(
        len(src_field.vocab),
        len(trg_field.vocab),
        d_model = d_model,
        n = 6,
        heads = 8,
        dropout = 0.1
    )

    n_gpu = torch.cuda.device_count()
    multiple_gpu = n_gpu > 1
    is_cuda = ('cuda' in device)
    if (multiple_gpu and is_cuda):
        print(f"Use {n_gpu} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

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
        0.2, 
        d_model, 
        4000
    )

    print("Creating loss function")
    criterion = LabelSmoothingLoss(
        len(trg_field.vocab), 
        trg_pad_idx = trg_pad, 
        smoothing = 0.1
    )

    print("Training begin")
    for epoch in range(n_epoch):
        total_loss = 0
        
        n_iter = 0
        for i, batch in enumerate(train_dataset): 
            loss = step(model, opt, batch, criterion, src_pad, trg_pad)
            n_iter += 1
            total_loss += loss
            if ((i+1)%print_every==0):
                avg_loss = total_loss / n_iter
                total_loss = 0
                n_iter = 0
                print("epoch: {:03d} - iter: {:04d} - loss: {:.4f}".format(epoch, i+1, avg_loss))

        valid_loss = validiate(model, val_dataset, criterion, src_pad, trg_pad)
        print('epoch: {:03d} - valid loss: {:.4f}'.format(epoch, valid_loss))
        torch.save(model.state_dict(), f'../models/model.{epoch}.pt')

if __name__=="__main__":
    main()