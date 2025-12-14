import pytest
from pydantic import ValidationError
from app.schemas import ProcessedHousePriceInputData, HousePriceOutputData, dummy_input

def test_processed_input_schema_valid():
        data = ProcessedHousePriceInputData(**dummy_input)
        assert isinstance(data, ProcessedHousePriceInputData)
        assert data.LotArea == 8450

def test_processed_input_schema_invalid():
    invalid_input = dummy_input.copy()
    invalid_input["LotArea"] = "string_instead_of_int"
    with pytest.raises(ValidationError):
        ProcessedHousePriceInputData(**invalid_input)

def test_missing_field_raises_error():
    missing_input = dummy_input.copy()
    missing_input.pop("LotArea")
    with pytest.raises(ValidationError):
        ProcessedHousePriceInputData(**missing_input)

def test_alias_fields_work():
     data = ProcessedHousePriceInputData(**dummy_input)
     assert data.FirstFlrSF == dummy_input["1stFlrSF"]
     assert data.SecondFlrSF == dummy_input["2ndFlrSF"]
     assert data.ThreeSsnPorch == dummy_input["3SsnPorch"]

def test_output_model_valid():
     out = HousePriceOutputData(SalePrice=250000.0)
     assert out.SalePrice == 250000.0  

def test_output_model_invalid():
        with pytest.raises(ValidationError):
            HousePriceOutputData(SalePrice="not_a_float")




