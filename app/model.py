import joblib
from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np


MODEL_PATH = Path(hf_hub_download(
    repo_id="k0stro/house-prices-model",
    filename="house_prices_model.pkl",
    repo_type="model"
))

def load_model():
    """Load the pre-trained model from Hugging Face."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

def predict(features: list[float]) -> float:
    """Make a prediction using the trained model."""
    X = np.array(features).reshape(1, -1)  # 1 sample, n_features
    y_pred = model.predict(X)
    return float(y_pred[0])