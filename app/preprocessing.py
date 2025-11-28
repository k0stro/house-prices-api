import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

def feature_construction(df: pd.DataFrame) -> pd.DataFrame:
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
    scaled_array = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df