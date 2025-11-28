from fastapi import FastAPI, Body
from app.schemas import RawHousePriceInputData, HousePriceOutputData, sample_raw_data
from app.model import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=HousePriceOutputData)
def predict_price(input_data: RawHousePriceInputData = Body(..., example=sample_raw_data)):
    features = list(input_data.model_dump().values())
    prediction = predict(features)
    return HousePriceOutputData(SalePrice=prediction)