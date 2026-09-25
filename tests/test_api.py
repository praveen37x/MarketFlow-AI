"""
Tests for FastAPI Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MarketFlow-AI"


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "DuckDB" in data["engine"]


def test_ai_predict_endpoint(client: TestClient):
    response = client.get("/api/v1/ai/predict?symbol=NIFTY")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NIFTY"
    assert "predicted_regime" in data
    assert "confidence" in data


def test_ai_copilot_endpoint(client: TestClient):
    response = client.get("/api/v1/ai/copilot?symbol=NIFTY")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "NIFTY"
    assert "market_outlook" in data
    assert "recommendation" in data
