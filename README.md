# House Price Prediction API

FastAPI + ML model serving in Docker. This project exposes a REST API that predicts house prices based on input features.

---

## Requirements
- [Docker](https://docs.docker.com/get-docker/) installed on your system

---

## Run with Docker

### 1. Clone the repository

```bash
git clone https://github.com/k0stro/house-prices-api.git
cd house-prices-api
```

### 2. Build the Docker image

```bash
docker build -t house-prices-api .
```

### 3. Run the container

```bash
docker run -d -p 8000:8000 house-prices-api
```

Now the API will available at http://localhost:8000/

---

## API Endpoints

### Health check

```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "ok"}
```

### Prediction
Send a POST request to ```/predict``` with JSON body containing the features:
```bash
curl -X POST http://localhost:8000/predict \ 
-H "Content-Type: application/json" \ 
-d '{ 
"LotFrontage": 60.0, 
"LotArea": 8450, 
"OverallQual": 7, 
"OverallCond": 5, 
"YearBuilt": 2003, 
"YearRemodAdd": 2003, 
"MasVnrArea": 196.0, 
"BsmtFinSF1": 706, 
... 
"AreaPerRoom": 146.5 
}'
```
Response example:
```json
{"SalePrice": 201121.37}
```

---
## Running tests

If you want to run unit test locally:
```bash
pytest -v
```
---
## Notes
- All dependencies are intalled inside the Docker image
- The trained model is loaded from Hugging Face
