from fastapi.testclient import TestClient
from app.main import app
from app.schemas import sample_raw_data

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json=sample_raw_data)
    assert response.status_code == 200
    data = response.json()
    assert "SalePrice" in data
    assert isinstance(data["SalePrice"], float)