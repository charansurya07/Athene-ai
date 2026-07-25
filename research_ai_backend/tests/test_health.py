"""
Smoke test for the health endpoint. Does not require any API keys —
run with: pytest
"""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["orchestrator"] == "online"


def test_root_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()
