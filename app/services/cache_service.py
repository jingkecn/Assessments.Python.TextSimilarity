from typing import Optional

from app.models import SimilarityMetric


class CacheService:
    """
    A caching service for similarity calculations.

    Cache strategy:
    - Similarity: (metric, text1 hash, text2 hash) -> similarity score

    TODO: Redis integration for persistent caching.
    """

    def __init__(self):
        self.similarity_cache = {}

    @staticmethod
    def _generate_similarity_key(metric: SimilarityMetric, text1: str, text2: str) -> str:
        """Generate a unique key for similarity based on texts and metric."""
        hash1, hash2 = hash(text1), hash(text2)
        if hash1 > hash2:
            hash1, hash2 = hash2, hash1  # Ensure consistent ordering
        # Use a consistent format for the key
        return f"sim:{metric}:{hash1}:{hash2}"

    def get_similarity(self, metric: str, text1: str, text2: str) -> Optional[float]:
        """Retrieve similarity score from cache or return None if not found."""
        key = self._generate_similarity_key(metric, text1, text2)
        return self.similarity_cache.get(key)

    def set_similarity(self, metric: str, text1: str, text2: str, score: float):
        """Store similarity score in cache."""
        key = self._generate_similarity_key(metric, text1, text2)
        self.similarity_cache[key] = score
