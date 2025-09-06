import pytest
from app.model import load_model, MODEL_PATH

def test_load_model():
    model = load_model()
    assert model is not None
    assert hasattr(model, 'predict')
