from fastapi import FastAPI
from app.schemas import HousePriceInputData, HousePriceOutputData
from app.model import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=HousePriceOutputData)
def predict_price(input_data: HousePriceInputData):
    pass