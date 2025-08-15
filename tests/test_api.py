from starlette.testclient import TestClient

from app.main import app
from app.utils.config import settings

client = TestClient(app)


class TestAPI:
    def test_endpoint_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == settings.SERVICE_NAME
        assert data["status"] == "healthy"
        assert data["version"] == settings.VERSION
