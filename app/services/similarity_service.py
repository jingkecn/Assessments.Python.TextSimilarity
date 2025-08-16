from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import SimilarityMetric
from app.services.cache_service import CacheService
from app.utils.config import settings


class TextSimilarityService:
    def __init__(self, cache_service: Optional[CacheService] = None):
        try:
            self.cache_service = cache_service
            # Initialize semantic similarity model
            self.semantic_model = SentenceTransformer(
                settings.SENTENCE_TRANSFORMER_MODEL,
                cache_folder=f"{Path.home()}/.cache/sentence_transformers"
            )
        except Exception as e:
            print(f"Failed to load semantic model: {e}")
            self.semantic_model = None

    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors."""
        similarity = self.cache_service.get_similarity(
            SimilarityMetric.COSINE.value, text1, text2) if self.cache_service else None
        if similarity is not None:
            return similarity

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

            if self.cache_service:
                self.cache_service.set_similarity(SimilarityMetric.COSINE.value, text1, text2, similarity)

            return similarity
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word sets."""
        similarity = self.cache_service.get_similarity(
            SimilarityMetric.JACCARD.value, text1, text2) if self.cache_service else None
        if similarity is not None:
            return similarity

        try:
            # Convert to lowercase and split into words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            if union == 0.0:
                return 0.0

            similarity = float(intersection / union)

            if self.cache_service:
                self.cache_service.set_similarity(SimilarityMetric.JACCARD.value, text1, text2, similarity)

            return similarity
        except Exception as e:
            print(f"Error calculating Jaccard similarity: {e}")
            return 0.0

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if self.semantic_model is None:
            print("Semantic model not available, falling back to cosine similarity")
            return self.cosine_similarity_tfidf(text1, text2)

        similarity = self.cache_service.get_similarity(
            SimilarityMetric.SEMANTIC.value, text1, text2) if self.cache_service else None
        if similarity is not None:
            return similarity

        try:
            # Get embeddings
            embeddings = self.semantic_model.encode([text1, text2])

            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            if self.cache_service:
                self.cache_service.set_similarity(SimilarityMetric.SEMANTIC.value, text1, text2, similarity)

            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return self.cosine_similarity_tfidf(text1, text2)

    def calculate_similarity(self, text1: str, text2: str, metric: SimilarityMetric) -> float:
        """Calculate similarity using specified metric."""
        metric_map = {
            SimilarityMetric.COSINE: self.cosine_similarity_tfidf,
            SimilarityMetric.JACCARD: self.jaccard_similarity,
            SimilarityMetric.SEMANTIC: self.semantic_similarity
        }

        similarity_func = metric_map.get(metric)
        return similarity_func(text1, text2)
