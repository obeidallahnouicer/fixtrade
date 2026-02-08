"""
Basic application tests.

Validates that the FastAPI app starts correctly and the
health endpoint responds as expected.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self) -> None:
        """Health endpoint must return HTTP 200 with status ok."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_body(self) -> None:
        """Health endpoint must return status and version fields."""
        response = client.get("/api/v1/health")
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body
