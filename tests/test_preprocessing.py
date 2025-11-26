import pytest
import pandas as pd
import numpy as np
from app.preprocessing import feature_construction, apply_ordinal_encoding, split_columns, cast_categorical
from app.schemas import sample_raw_data

@pytest.fixture
def sample_df():
    return pd.DataFrame([sample_raw_data])

def test_feature_construction():
    data = {
        "YrSold": [2020, 2010],
        "YearBuilt": [2000, 2005],
        "YearRemodAdd": [2005, 2008],
        "GarageType": ["Attchd", "None"],
        "TotalBsmtSF": [800, 0],
        "BsmtUnfSF": [200, 0],
        "GrLivArea": [1500, 1200],
        "LotArea": [9000, 10000],
        "1stFlrSF": [1000, 900],
        "2ndFlrSF": [500, 300],
        "TotRmsAbvGrd": [6, 5],
    }
    df = pd.DataFrame(data)
    result = feature_construction(df.copy())

    assert "Age" in result.columns
    assert "RemodAge" in result.columns
    assert "HasGarage" in result.columns
    assert "HasBasement" in result.columns
    assert "UnfBsmtPercent" in result.columns
    assert "LivLotRatio" in result.columns
    assert "AreaPerRoom" in result.columns

    assert result.loc[0, "Age"] == 20  # 2020 - 2000
    assert result.loc[0, "RemodAge"] == 15  # 2020 - 2005
    assert result.loc[0, "HasGarage"] == 1  # bo GarageType != 'None'
    assert result.loc[1, "HasGarage"] == 0  # bo GarageType == 'None'
    assert result.loc[0, "HasBasement"] == 1
    assert result.loc[1, "HasBasement"] == 0
    assert np.isclose(result.loc[0, "UnfBsmtPercent"], 0.25)  # 200 / 800
    assert result.loc[1, "UnfBsmtPercent"] == 0
    assert np.isclose(result.loc[0, "LivLotRatio"], 1500 / 9000)
    assert np.isclose(result.loc[0, "AreaPerRoom"], (1000 + 500) / 6)


def test_apply_ordinal_target_encoding():
    data = {
        "Neighborhood": ["A", "B", "A", "C"],
        "HouseStyle": ["1Story", "2Story", "1Story", "1Story"],
    }
    df = pd.DataFrame(data)

    encoding_map = {
        "Neighborhood": {"A": 1, "B": 2, "C": 3},
        "HouseStyle": {"1Story": 1, "2Story": 2},
    }

    result = apply_ordinal_encoding(df.copy(), encoding_map)

    assert result.loc[0, "Neighborhood_encoded"] == 1
    assert result.loc[1, "Neighborhood_encoded"] == 2
    assert result.loc[3, "Neighborhood_encoded"] == 3
    assert result.loc[0, "HouseStyle_encoded"] == 1
    assert result.loc[1, "HouseStyle_encoded"] == 2

def test_split_columns(sample_df):
    numerical_cols, categorical_cols = split_columns(sample_df)

    expected_numerical_cols = [
        'LotFrontage',
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'MasVnrArea',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        '1stFlrSF',
        '2ndFlrSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'GarageYrBlt',
        'GarageCars',
        'GarageArea',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        '3SsnPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'MoSold',
        'YrSold'
        ]

    expected_categorical_cols = [
        'MSZoning',
        'Street',
        'Alley',
        'LotShape',
        'LandContour',
        'Utilities',
        'LotConfig',
        'LandSlope',
        'Neighborhood',
        'Condition1',
        'Condition2',
        'BldgType',
        'HouseStyle',
        'RoofStyle',
        'RoofMatl',
        'Exterior1st',
        'Exterior2nd',
        'MasVnrType',
        'ExterQual',
        'ExterCond',
        'Foundation',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'Heating',
        'HeatingQC',
        'CentralAir',
        'Electrical',
        'KitchenQual',
        'Functional',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'PavedDrive',
        'PoolQC',
        'Fence',
        'MiscFeature',
        'SaleType',
        'SaleCondition',
        'MSSubClass'
        ]

    assert set(numerical_cols) == set(expected_numerical_cols)
    assert set(categorical_cols) == set(expected_categorical_cols)

def test_cast_categorical_changes_dtype(sample_df):
    categorical_cols = [
        'MSZoning',
        'Street',
        'Alley',
        'LotShape',
        'LandContour',
        'Utilities',
        'LotConfig',
        'LandSlope',
        'Neighborhood',
        'Condition1',
        'Condition2',
        'BldgType',
        'HouseStyle',
        'RoofStyle',
        'RoofMatl',
        'Exterior1st',
        'Exterior2nd',
        'MasVnrType',
        'ExterQual',
        'ExterCond',
        'Foundation',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'Heating',
        'HeatingQC',
        'CentralAir',
        'Electrical',
        'KitchenQual',
        'Functional',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'PavedDrive',
        'PoolQC',
        'Fence',
        'MiscFeature',
        'SaleType',
        'SaleCondition',
        'MSSubClass'
        ]
    result = cast_categorical(sample_df, categorical_cols)

    for col in categorical_cols:
        assert result[col].dtype == 'object'