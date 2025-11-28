from fastapi import FastAPI, Body
from app.schemas import RawHousePriceInputData, HousePriceOutputData, dummy_input
from app.model import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=HousePriceOutputData)
def predict_price(input_data: RawHousePriceInputData = Body(..., example=dummy_input)):
    features = list(input_data.model_dump().values())
    prediction = predict(features)
    return HousePriceOutputData(SalePrice=prediction)