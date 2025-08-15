from locust import HttpUser, task


class TextSimilarityUser(HttpUser):
    @task(1)
    def test_health_check(self):
        """Test health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Service not healthy: {data.get('status')}")
            else:
                response.failure(f"Health check failed: status code={response.status_code}")
