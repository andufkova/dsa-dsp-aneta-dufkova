import pandas as pd

def continuous_data_without_na(df):
    data_num = df.select_dtypes(include='number')
    data = data_num.dropna()
    return data

def categorical_data(df):
    return df.select_dtypes(include='dtype')

def categorical_data_without_na(df, na_cols):
    data_drop = df.drop(na_cols, axis=1)
    data_na = data_drop.dropna()
    return data_na

def one_hot_encode(df, enc):
    df_tmp = pd.DataFrame(df.index)
    df_tmp = df_tmp.set_index(['Id'])
    column_name = enc.get_feature_names(df.columns)
    encoded = pd.DataFrame(enc.transform(df).toarray(), columns=column_name)
    encoded = encoded.set_index(df_tmp.index.copy())
    return encoded

