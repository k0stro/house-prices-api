import joblib
from pathlib import Path
from pydantic import BaseModel, Field

class HousePriceData(BaseModel):
    """Data model for house price prediction input."""
    LotFrontage: float
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: float
    BsmtFinSF1: int
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    FirstFlrSF: int = Field(alias='1stFlrSF')
    SecondFlrSF: int = Field(alias='2ndFlrSF')
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    TotRmsAbvGrd: int
    Fireplaces: int
    GarageYrBlt: float
    GarageCars: int
    GarageArea: int
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int = Field(alias='3SsnPorch')
    ScreenPorch: int
    PoolArea: int
    MiscVal: int
    MoSold: int
    YrSold: int
    MSZoning_encoded: int
    Street_encoded: int
    Alley_encoded: int
    LotShape_encoded: int
    LandContour_encoded: int
    Utilities_encoded: int
    LotConfig_encoded: int
    LandSlope_encoded: int
    Neighborhood_encoded: int
    Condition1_encoded: int
    Condition2_encoded: int
    BldgType_encoded: int
    HouseStyle_encoded: int
    RoofStyle_encoded: int
    RoofMatl_encoded: int
    Exterior1st_encoded: int
    Exterior2nd_encoded: int
    MasVnrType_encoded: int
    ExterQual_encoded: int
    ExterCond_encoded: int
    Foundation_encoded: int
    BsmtQual_encoded: int
    BsmtCond_encoded: int
    BsmtExposure_encoded: int
    BsmtFinType1_encoded: int
    BsmtFinType2_encoded: int
    Heating_encoded: int
    HeatingQC_encoded: int
    CentralAir_encoded: int
    Electrical_encoded: int
    KitchenQual_encoded: int
    Functional_encoded: int
    FireplaceQu_encoded: int
    GarageType_encoded: int
    GarageFinish_encoded: int
    GarageQual_encoded: int
    GarageCond_encoded: int
    PavedDrive_encoded: int
    PoolQC_encoded: int
    Fence_encoded: int
    MiscFeature_encoded: int
    SaleType_encoded: int
    SaleCondition_encoded: int
    MSSubClass_encoded: int
    Age: int
    RemodAge: int
    HasGarage: int
    HasBasement: int
    UnfBsmtPercent: float
    LivLotRatio: float
    AreaPerRoom: float


MODEL_PATH = Path('models/house_prices_model.pkl')

def load_model():
    """Load the pre-trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

