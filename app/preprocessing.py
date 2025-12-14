import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from pydantic import BaseModel

def feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    if '1stFlrSF' not in df.columns and 'FirstFlrSF' in df.columns:
        df['1stFlrSF'] = df['FirstFlrSF']
    if '2ndFlrSF' not in df.columns and 'SecondFlrSF' in df.columns:
        df['2ndFlrSF'] = df['SecondFlrSF']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasGarage'] = (df['GarageType'] != 'None').astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['UnfBsmtPercent'] = np.where(df['TotalBsmtSF'] != 0, df['BsmtUnfSF'] / df['TotalBsmtSF'], 0)
    df['LivLotRatio'] = df['GrLivArea'] / df['LotArea']
    df['AreaPerRoom'] = (df['1stFlrSF'] + df['2ndFlrSF'])  / df['TotRmsAbvGrd']
    return df

def apply_ordinal_encoding(df: pd.DataFrame, encoding_map: dict) -> pd.DataFrame:
    """Applies a saved target-guided ordinal encoding map to a new DataFrame

    Parameters:
    df: pandas DataFrame
    encoding_map: dict, saved encoding maps {column_name: {category: encoded_value}}
    Returns:
    df: DataFrame with encoded columns
    """
    df = df.copy()
    for col, col_map in encoding_map.items():
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map(col_map)
    return df

def split_columns(df: pd.DataFrame, target_col: str = 'SalePrice', id_col: str = 'Id') -> tuple[list[str], list[str]]:
    """
    Splits dataframe columns into numerical and categorical
    with MSSubClass treated as categorical.

    Parameters:
    df: pandas DataFrame
    target_col: str, name of the target column to exclude
    id_col: str, name of the ID column to exclude
    Returns:
    tuple: (numerical_cols, categorical_cols)
    """
    numerical_cols = [col for col in df.columns if df.dtypes[col] != 'object']
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if id_col in numerical_cols:
        numerical_cols.remove(id_col)

    categorical_cols = [col for col in df.columns if df.dtypes[col] == 'object']

    if 'MSSubClass' in numerical_cols:
        numerical_cols.remove('MSSubClass')
        categorical_cols.append('MSSubClass')

    return numerical_cols, categorical_cols

def cast_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[categorical_cols] = df[categorical_cols].astype('object')
    return df

def drop_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=categorical_cols)
    return df

def scale_features(df: pd.DataFrame, scaler: ColumnTransformer) -> pd.DataFrame:
    df = df.copy()
    scaled_array = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df

def preprocess_data(raw_data: BaseModel, encoding_map: dict, scaler: ColumnTransformer, train_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([raw_data.model_dump(by_alias=True)])
    _, categorical_cols = split_columns(df)
    df = cast_categorical(df, categorical_cols)
    df_encoded = apply_ordinal_encoding(df, encoding_map)
    df_encoded = feature_construction(df_encoded)
    df_encoded = drop_categorical(df_encoded, categorical_cols)
    df_encoded = df_encoded[train_columns]
    df_scaled = scale_features(df_encoded, scaler)
    return df_scaled