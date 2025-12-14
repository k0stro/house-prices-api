from fastapi import FastAPI, Body
from app.schemas import RawHousePriceInputData, HousePriceOutputData, sample_raw_data, TRAIN_COLUMNS
from app.model import predict, load_encoding_map, load_scaler
from app.preprocessing import preprocess_data

app = FastAPI()
encoding_map = load_encoding_map()
scaler = load_scaler()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=HousePriceOutputData)
def predict_price(input_data: RawHousePriceInputData = Body(..., example=sample_raw_data)):
    input_preprocess = preprocess_data(input_data, encoding_map=encoding_map, scaler=scaler, train_columns=TRAIN_COLUMNS)
    prediction = predict(input_preprocess)
    return HousePriceOutputData(SalePrice=prediction)