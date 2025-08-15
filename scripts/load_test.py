import random

from locust import HttpUser, task


class TextSimilarityUser(HttpUser):
    def on_start(self) -> None:
        """Called when a simulated user starts."""
        # Test data for different scenarios
        self.test_prompts = [
            ("What is agentic AI?", "Explain AI agents."),
            ("The weather is nice today", "it's a beautiful sunny day"),
            ("How to cook pasta?", "What is the recipe for spaghetti?"),
            ("Python programming tutorial", "Learn Python coding"),
            ("Database design principles", "SQL database optimization"),
            ("Climate change effects", "Global warming consequences"),
            ("Travel to Paris", "Visit France capital city"),
            ("Healthy eating habits", "Nutritious food choices"),
            ("Exercise routine", "Fitness workout plan"),
            ("Book recommendations", "Suggest good novels to read")
        ]

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

    @task(1)
    def test_available_metrics(self):
        """Test metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "available_metrics" in data:
                    response.success()
                else:
                    response.failure("Metrics not available")
            else:
                response.failure(f"Metrics endpoint failed: status code={response.status_code}")

    @task(3)
    def test_similarity_cosine(self):
        """Test similarity endpoint with cosine metric."""
        prompt1, prompt2 = random.choice(self.test_prompts)

        payload = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "similarity_metric": "cosine",
            "similarity_threshold": random.uniform(0.5, 0.9)
        }

        with self.client.post("/similarity", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Similarity endpoint failed: status code={response.status_code}")

    @task(1)
    def test_similarity_jaccard(self):
        """Test similarity endpoint with Jaccard metric."""
        prompt1, prompt2 = random.choice(self.test_prompts)

        payload = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "similarity_metric": "jaccard",
            "similarity_threshold": random.uniform(0.4, 0.8)
        }

        with self.client.post("/similarity", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Similarity endpoint failed: status code={response.status_code}")

    @task(2)
    def test_similarity_semantic(self):
        """Test similarity endpoint with semantic metric."""
        prompt1, prompt2 = random.choice(self.test_prompts)

        payload = {
            "prompt1": prompt1,
            "prompt2": prompt2,
            "similarity_metric": "semantic",
            "similarity_threshold": random.uniform(0.3, 0.7)
        }

        with self.client.post("/similarity", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Similarity endpoint failed: status code={response.status_code}")
