from torchtext import data
import os
from utils.loader import MyIterator, batch_size_fn

def create_dataset(df, src_field, trg_field, batch_size, max_strlen, device, istrain=True):

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
        batch_size = batch_size, 
        device = device,
        repeat = False, 
        sort_key = lambda x: (len(x.src), len(x.trg)),
        train = istrain, 
        shuffle = True,
        batch_size_fn = batch_size_fn
    )
    return iterator