from torchtext.data import field

def create_field(src_tokenizer, trg_tokenizer):
    src_field = field.Field(
        lower = True, 
        tokenize = src_tokenizer.tokenize
    )
    trg_field = field.Field(
        lower = True, 
        tokenize = trg_tokenizer.tokenize, 
        init_token = '<sos>', 
        eos_token = '<eos>'
    )
    return src_field, trg_field