import pandas as pd
import copy
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load

# During training, model.feature_importances_ method was used to get only the most important features
L_IMPORTANT = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
       'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL',
       'MSZoning_RM', 'Street_Grvl', 'Street_Pave', 'LotShape_IR1',
       'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'LandContour_Bnk',
       'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl',
       'Utilities_AllPub', 'Utilities_NoSeWa', 'LotConfig_Corner',
       'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3',
       'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev',
       'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale',
       'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr',
       'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert',
       'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel',
       'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes',
       'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown',
       'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW',
       'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber',
       'Neighborhood_Veenker', 'Condition1_Artery', 'Condition1_Feedr',
       'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN',
       'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe',
       'Condition1_RRNn', 'Condition2_Artery', 'Condition2_Feedr',
       'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN',
       'Condition2_RRAe']

def continuous_data_without_na(df):
    data_na = df.dropna()
    return data_na

def continuous_data(df):
    data_num = df.select_dtypes(include='number')
    return data_num

def categorical_data(df):
    data_cat = df.select_dtypes(include='dtype')
    return data_cat

def categorical_data_without_na(df):
    data_na = df.dropna()
    return data_na

def select_categorical(df, cols=None):
    if cols is None:
        cat_cols = df.columns[df.isna().any()].tolist()
    else:
        cat_cols = cols
    data_drop = df.drop(cat_cols, axis=1)
    return data_drop, cat_cols

def select_continuous(df, target_column, cols=None):
    if cols is None:
        highest_corr = df.corrwith(df[target_column]).sort_values(ascending=False)
        con_data_cols = list(highest_corr[1:6].index)

        top_correlated_target = copy.deepcopy(con_data_cols)
        top_correlated_target.append(target_column)
        top_correlated_target = set(top_correlated_target)
    else:
        top_correlated_target = cols
        con_data_cols = cols

    data = df[top_correlated_target]
    return data, con_data_cols

def one_hot_encode(df, enc):
    df_tmp = pd.DataFrame(df.index)
    df_tmp = df_tmp.set_index(['Id'])
    column_name = enc.get_feature_names(df.columns)
    encoded = pd.DataFrame(enc.transform(df).toarray(), columns=column_name)
    encoded = encoded.set_index(df_tmp.index.copy())
    return encoded

def fill_na(data, method='zero'):
    if method == 'zero':
        return data.fillna(0)
    else:
        return data.fillna(data.mode().iloc[0])

def preprocess_data(df, target_column, cols_con=None, cols_cat=None):
    df_continuous = continuous_data(df)
    df_categorical = categorical_data(df)
    df_continuous_selected, cols_con = select_continuous(df_continuous, target_column, cols_con)
    df_categorical_selected, cols_cat = select_categorical(df_categorical, cols_cat)
    if cols_con is None:
        df_continuous_without_na = continuous_data_without_na(df_continuous_selected)
        df_categorical_without_na = categorical_data_without_na(df_categorical_selected)
    else:
        df_continuous_without_na = fill_na(df_continuous_selected)
        df_categorical_without_na = fill_na(df_categorical_selected, 'mode')
    return df_continuous_without_na, df_categorical_without_na, cols_con, cols_cat

def extract_features(enc, contin, categor):
    categ_final = one_hot_encode(categor, enc)
    return contin, categ_final

def select_features(df_final):
    df_final.drop(L_IMPORTANT, axis=1)
    return df_final

def prepare_data(df, target_column, ROOT_DIR):
    contin, categor, cols_con, cols_cat = preprocess_data(df, target_column)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categor)
    dump(enc, ROOT_DIR / 'models' / 'encoder.joblib')
    contin_final, categ_final = extract_features(enc, contin, categor)
    
    data = pd.merge(contin_final, categ_final, left_index=True, right_index=True)
    data_final = select_features(data)
    return enc, data_final, cols_con, cols_cat

def prepare_data_inference(df, ROOT_DIR, cols_con, cols_cat):
    contin, categor, cols_con, cols_cat = preprocess_data(df, '', cols_con, cols_cat)
    enc = load(ROOT_DIR / 'models' / 'encoder.joblib')
    contin_final, categ_final = extract_features(enc, contin, categor)
    
    data = pd.merge(contin_final, categ_final, left_index=True, right_index=True)
    data_final = select_features(data)
    return data_final

