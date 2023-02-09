import spacy
import re

class tokenizer(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
        
    def tokenize(self, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]",
            " ",
            sentence
        )
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        
        tokens = []
        for tok in self.nlp.tokenizer(sentence):
            if (tok.text != " "): 
                tokens.append(tok.text)
        return tokens

if __name__=="__main__":
    
    sentence = "Xin chào bạn, tui là Phúc"
    vi_tokenizer = tokenizer('vi_core_news_lg')
    print(f"{sentence} -> {vi_tokenizer.tokenize(sentence)}")

    sentence = "Hello, I am Phuc"
    en_tokenizer = tokenizer('en_core_web_sm')
    print(f"{sentence} -> {en_tokenizer.tokenize(sentence)}")