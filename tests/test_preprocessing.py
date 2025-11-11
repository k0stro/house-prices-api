import pytest
import pandas as pd
import numpy as np
from app.preprocessing import feature_construction

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