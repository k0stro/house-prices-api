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