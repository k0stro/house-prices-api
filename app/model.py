import joblib
import numpy as np
from config.main_config import MODEL_PATH, ENCODING_MAP_PATH, SCALER_PATH

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

def load_encoding_map():
    """Load the categorical encoding map from Hugging Face."""
    if not ENCODING_MAP_PATH.exists():
        raise FileNotFoundError(f"Encoding map file not found at {ENCODING_MAP_PATH}")
    encoding_map = joblib.load(ENCODING_MAP_PATH)
    return encoding_map

encoding_map = load_encoding_map()

def load_scaler():
    """Load the column transformer scaler from Hugging Face."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    return scaler

scaler = load_scaler()