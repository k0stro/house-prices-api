import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from app.preprocessing import feature_construction, apply_ordinal_encoding, split_columns, cast_categorical, drop_categorical, preprocess_data, scale_features
from app.schemas import sample_raw_data, SampleModel, RawHousePriceInputData
from app.model import load_encoding_map, load_scaler
from app.schemas import TRAIN_COLUMNS

scaler = load_scaler()
encoding_map = load_encoding_map()
@pytest.fixture
def sample_df():
    return pd.DataFrame([sample_raw_data])
    
@pytest.fixture
def sample_model_for_pipeline():
    return RawHousePriceInputData(**sample_raw_data)

@pytest.fixture
def categorical_cols():
    cols = [
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
    return cols

@pytest.fixture
def sample_model_instance():
    return SampleModel()

@pytest.fixture
def mock_encoding_map():
    return {
        "Neighborhood": {"A": 1, "B": 2},
        "GarageType": {"Attchd": 1, "Detchd": 2}
    }

@pytest.fixture
def fitted_mock_scaler():
    df = pd.DataFrame({
        "GrLivArea": [1000, 2000],
        "LotArea": [5000, 10000],
        "Age": [10, 20],
        "RemodAge": [5, 10],
    })
    scaler = ColumnTransformer(
        transformers=[("minmax", MinMaxScaler(), ["GrLivArea", "LotArea", "Age", "RemodAge"])],
        remainder="passthrough"
    )
    scaler.fit(df)
    return scaler

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
    assert result.loc[0, "HasGarage"] == 1
    assert result.loc[1, "HasGarage"] == 0
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


def test_apply_ordinal_encoding_handles_missing_columns(sample_df):
    encoding_map = {
        "NonExistentColumn": {"A": 1, "B": 2},
    }

    result = apply_ordinal_encoding(sample_df.copy(), encoding_map)

    assert "NonExistentColumn_encoded" not in result.columns


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

def test_cast_categorical_changes_dtype(sample_df, categorical_cols):
    result = cast_categorical(sample_df, categorical_cols)

    for col in categorical_cols:
        assert result[col].dtype == 'object'

def test_cast_categorical_no_side_effects(sample_df, categorical_cols):
    original_df = sample_df.copy()
    result = cast_categorical(sample_df, categorical_cols)

    assert sample_df is not result
    assert sample_df.equals(original_df)


def test_drop_categorical_removes_columns(sample_df, categorical_cols):
    result = drop_categorical(sample_df, categorical_cols)

    for col in categorical_cols:
        assert col not in result.columns

def test_drop_categorical_keeps_other_columns(sample_df, categorical_cols):
    result = drop_categorical(sample_df, categorical_cols)

    expected_remaining_cols = set(sample_df.columns) - set(categorical_cols)

    assert set(result.columns) == expected_remaining_cols


def test_scale_features_applies_scaling():
    df = pd.DataFrame({
        'A': [0, 10],
        'B': [5, 15]
    })

    scaler = ColumnTransformer(
        transformers=[('scale', MinMaxScaler(), ['A', 'B'])],
        remainder='passthrough'
    )

    scaler.fit(df)

    result = scale_features(df, scaler)

    assert np.all(result['A'] >= 0) and np.all(result['A'] <= 1)
    assert np.all(result['B'] >= 0) and np.all(result['B'] <= 1)

def test_scale_features_preserves_shape_and_columns():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [10, 20, 30]
    })

    scaler = ColumnTransformer(
        transformers=[('scale', MinMaxScaler(), ['A', 'B'])],
        remainder='passthrough'
    )
    scaler.fit(df)

    result = scale_features(df, scaler)

    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)


def test_preprocess_data_full_pipeline(sample_model_for_pipeline):
    df_processed = preprocess_data(
        raw_data=sample_model_for_pipeline,
        encoding_map=encoding_map,
        scaler=scaler, train_columns=TRAIN_COLUMNS
    )

    assert isinstance(df_processed, pd.DataFrame)

    assert "Neighborhood" not in df_processed.columns
    assert "GarageType" not in df_processed.columns

    assert "Neighborhood_encoded" in df_processed.columns
    assert "GarageType_encoded" in df_processed.columns

    assert "Age" in df_processed.columns
    assert "RemodAge" in df_processed.columns
    assert "HasGarage" in df_processed.columns
    assert "HasBasement" in df_processed.columns

    assert df_processed.isna().sum().sum() == 0

    for col in ["GrLivArea", "LotArea", "Age", "RemodAge"]:
        assert df_processed[col].dtype == float

    #assert not df_processed.equals(original_df)