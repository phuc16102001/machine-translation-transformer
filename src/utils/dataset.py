from torchtext import data
from ..config.config import config
import os
from loader import MyIterator

def create_dataset(df, src_field, trg_field, istrain=True):
    max_strlen = config['max_strlen']

    # Filter empty data
    non_empty_src = (df['src'].str.count(' ') < max_strlen)
    non_empty_trg = (df['trg'].str.count(' ') < max_strlen)
    mask = non_empty_src & non_empty_trg
    df = df.loc[mask]
    df.to_csv('data_tmp.csv', index=False)
    
    data_fields = [
        ('src', src_field),
        ('trg', trg_field)
    ]    
    dataset = data.TabularDataset(
        './data_tmp.csv', 
        format = 'csv', 
        fields = data_fields
    )
    os.remove('data_tmp.csv')
    
    if (istrain):
        print("Create vocabulary...")
        src_field.build_vocab(dataset)
        trg_field.build_vocab(dataset)
        
    iterator = MyIterator(
        dataset, 
        batch_size = config['batch_size'], 
        device = config['device'],
        repeat = False, 
        sort_key = lambda x: (len(x.src), len(x.trg)),
        train = istrain, 
        shuffle = True
    )
    return iterator