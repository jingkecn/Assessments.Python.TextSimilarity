from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import SimilarityMetric


class TextSimilarityService:
    def __init__(self):
        try:
            # Initialize semantic similarity model
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Failed to load semantic model: {e}")
            self.semantic_model = None

    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word sets."""
        try:
            # Convert to lowercase and split into words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            if union == 0.0:
                return 0.0

            return intersection / union
        except Exception as e:
            print(f"Error calculating Jaccard similarity: {e}")
            return 0.0

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if self.semantic_model is None:
            print("Semantic model not available, falling back to cosine similarity")
            return self.cosine_similarity_tfidf(text1, text2)

        try:
            # Get embeddings
            embeddings = self.semantic_model.encode([text1, text2])

            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

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
