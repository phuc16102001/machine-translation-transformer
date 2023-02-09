import torch

def main():
    src = torch.load('../models/src_vocab.pth')
    trg = torch.load('../models/trg_vocab.pth')
    model = torch.load('../models/model_best.pth')

if __name__=="__main__":
    main()