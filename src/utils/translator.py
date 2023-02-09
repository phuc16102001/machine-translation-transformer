from config.config import config
from torch.autograd import Variable
import torch

def translate(sentence, model, src_field, trg_field):
    model.eval()

    max_strlen = config['max_strlen']
    device = config['device']

    indexed = []
    sentence = src_field.preprocess(sentence)
    
    for tok in sentence:
        if src_field.vocab.stoi[tok] != src_field.vocab.stoi['<eos>']:
            indexed.append(src_field.vocab.stoi[tok])
    
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(device)