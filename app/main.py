from fastapi import FastAPI
from app.schemas import ProcessedHousePriceInputData, HousePriceOutputData
from app.model import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=HousePriceOutputData)
def predict_price(input_data: ProcessedHousePriceInputData): #ProcessedHousePriceInputData to be swapped with RawProcessedHousePriceInputData for raw input after adding preprocessing step
    features = list(input_data.model_dump().values())
    prediction = predict(features)
    return HousePriceOutputData(SalePrice=prediction)