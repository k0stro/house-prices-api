import joblib
from pathlib import Path

MODEL_PATH = Path('models/house_prices_model.pkl')

def load_model():
    """Load the pre-trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model