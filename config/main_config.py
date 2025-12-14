from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "k0stro/house-prices-model"

MODEL_PATH = Path(hf_hub_download(
    repo_id=repo_id,
    filename="house_prices_model.pkl",
    repo_type="model"
))

ENCODING_MAP_PATH = Path(hf_hub_download(
    repo_id=repo_id,
    filename="categorical_encoding_map.pkl",
    repo_type="model"
))

SCALER_PATH = Path(hf_hub_download(
    repo_id=repo_id,
    filename="column_transformer_scaler.pkl",
    repo_type="model"
))