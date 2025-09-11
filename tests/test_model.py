import pytest
import joblib
from app.model import load_model, MODEL_PATH, predict

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