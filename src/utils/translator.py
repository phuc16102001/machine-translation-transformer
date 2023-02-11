from torch.autograd import Variable
import torch
import re
from utils.beam_search import beam_search

def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def convert_to_word(idx_list, vocab):
    word_list = []
    for tok in idx_list:
        word_list.append(vocab.itos[tok])
    return ' '.join(word_list)

def translate(sentence, model, src_field, trg_field, max_strlen, device, k):
    model.eval()

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
    sentence = sentence[1:sentence_len]
    sentence = convert_to_word(sentence, trg_field.vocab)
    sentence = multiple_replace({
        ' ?': '?',
        ' !': '!',
        ' .': '.',
        '\' ': '\'',
        ' ,':',',
        ' & apos ': '\''
    }, sentence)
    return sentence