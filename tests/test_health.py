"""Smoke test for the /health endpoint.

TestClient wraps the FastAPI app so tests call it in-process (no real HTTP
socket, no running server needed) - the FastAPI analog of Spring's
MockMvc/WebTestClient.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
