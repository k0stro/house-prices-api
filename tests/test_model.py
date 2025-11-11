import pytest
import joblib
from config.main_config import MODEL_PATH, ENCODING_MAP_PATH, SCALER_PATH
from app.model import   load_model, predict, load_encoding_map, load_scaler

def test_load_model():
    model = load_model()
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_path():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.suffix == '.pkl'

def test_predict_returns_float():
    dummy_features = [1.0 for _ in range(86)]
    result = predict(dummy_features)
    assert isinstance(result, float)

def test_load_encoding_map():
    encoding_map = load_encoding_map()
    assert encoding_map is not None
    assert isinstance(encoding_map, dict)

def test_encoding_map_path():
    assert ENCODING_MAP_PATH.exists()
    assert ENCODING_MAP_PATH.suffix == '.pkl'

def test_load_scaler():
    scaler = load_scaler()
    assert scaler is not None
    assert hasattr(scaler, 'transform')

def test_scaler_path():
    assert SCALER_PATH.exists()
    assert SCALER_PATH.suffix == '.pkl'