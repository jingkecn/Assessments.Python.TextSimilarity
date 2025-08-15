from unittest.mock import patch, AsyncMock

from starlette.testclient import TestClient

from app.main import app
from app.models import SimilarityMetric
from app.utils.config import settings

client = TestClient(app)


class TestAPI:
    def test_endpoint_health(self):
        with patch("app.main.llm_service") as mock_llm:
            mock_llm.is_available = AsyncMock(return_value=True)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["is_llm_available"] == True
            assert data["service"] == settings.SERVICE_NAME
            assert data["status"] == "healthy"
            assert data["version"] == settings.VERSION

    def test_endpoint_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "available_metrics" in data
        assert SimilarityMetric.COSINE in data["available_metrics"]
        assert SimilarityMetric.JACCARD in data["available_metrics"]
        assert SimilarityMetric.SEMANTIC in data["available_metrics"]

    def test_endpoint_similarity_basic(self):
        payload = {
            "prompt1": "This is a test sentence.",
            "prompt2": "This is another test sentence.",
            "similarity_metric": "cosine",
            "similarity_threshold": 0.5,
            "use_llm": True
        }

        with (
            patch("app.main.llm_service") as mock_llm,
            patch("app.main.similarity_service") as mock_sim
        ):
            # Mock LLM service
            mock_llm.generate_response = AsyncMock(return_value="This is a test response")

            # Mock similarity service
            mock_sim.calculate_similarity.return_value = 0.8

            response = client.post("/similarity", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["are_similar"] is True
            assert data["similarity_score"] == 0.8
            assert "llm_response" in data

    def test_endpoint_similarity_validation_errors(self):
        with (
            patch("app.main.llm_service"),
            patch("app.main.similarity_service")
        ):
            # Test empty prompt
            payload = {
                "prompt1": "",
                "prompt2": "Valid prompt",
                "similarity_metric": "cosine"
            }
            response = client.post("/similarity", json=payload)
            assert response.status_code == 422  # Validation error

            # Test invalid similarity metric
            payload = {
                "prompt1": "Valid prompt",
                "prompt2": "Another valid prompt",
                "similarity_metric": "invalid_metric"
            }
            response = client.post("/similarity", json=payload)
            assert response.status_code == 422

            # Test invalid threshold
            payload = {
                "prompt1": "Valid prompt",
                "prompt2": "Another valid prompt",
                "similarity_metric": "cosine",
                "similarity_threshold": 1.5  # > 1.0
            }
            response = client.post("/similarity", json=payload)
            assert response.status_code == 422
