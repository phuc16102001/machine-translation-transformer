import pandas as pd

def create_data(src_path, trg_path):
    src_data = open(src_path, encoding='utf8').read().strip().split('\n')
    trg_data = open(trg_path, encoding='utf8').read().strip().split('\n')

    df = pd.DataFrame({
        'src': [line for line in src_data],
        'trg': [line for line in trg_data]
    })
    not_empty = ((df['src'].apply(len)>0) & (df['trg'].apply(len)>0))
    df = df[not_empty]
    return df

if __name__=="__main__":
    df_train = create_data('data/train.vi', 'data/train.en')
    df_val = create_data('data/tst2013.vi', 'data/tst2013.en')

    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")
    print("="*10,"Sample train data","="*10)
    print(df_train.head())