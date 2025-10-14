from fastapi.testclient import TestClient
from app.main import app
from app.schemas import dummy_input

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json=dummy_input)
    assert response.status_code == 200
    data = response.json()
    assert "SalePrice" in data
    assert isinstance(data["SalePrice"], float) 