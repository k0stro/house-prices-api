import pandas as pd
import numpy as np
from app.model import load_encoding_map, load_scaler

def feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['HasGarage'] = (df['GarageType'] != 'None').astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['UnfBsmtPercent'] = np.where(df['TotalBsmtSF'] != 0, df['BsmtUnfSF'] / df['TotalBsmtSF'], 0)
    df['LivLotRatio'] = df['GrLivArea'] / df['LotArea']
    df['AreaPerRoom'] = (df['1stFlrSF'] + df['2ndFlrSF'])  / df['TotRmsAbvGrd']
    return df