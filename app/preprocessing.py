import pandas as pd
import numpy as np

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
    df_encoded: DataFrame with encoded columns
    """
    df = df.copy()
    for col, col_map in encoding_map.items():
        if col in df.columns:
            df[f'{col}_encoded'] = df[col].map(col_map)
    return df

def split_columns():
    pass
