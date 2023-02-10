from config.config import config
from torch.autograd import Variable
import torch
import re
from utils.beam_search import beam_search

def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate(sentence, model, src_field, trg_field):
    model.eval()

    max_strlen = config['max_strlen']
    device = config['device']
    k = config['k']

    indexed = []
    sentence = src_field.preprocess(sentence)
    
    for tok in sentence:
        indexed.append(src_field.vocab.stoi[tok])
    
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = sentence.to(device)
    sentence, sentence_len = beam_search(
        sentence, 
        model,
        src_field,
        trg_field,
        device,
        k,
        max_strlen
    )
    sentence = ' '.join([trg_field.vocab.itos[tok] for tok in sentence[1: sentence_len]])
    sentence = multiple_replace({
        ' ?': '?',
        ' !': '!',
        ' .': '.',
        '\' ': '\'',
        ' ,':',',
        ' & apos ': '\''
    }, sentence)
    return sentence