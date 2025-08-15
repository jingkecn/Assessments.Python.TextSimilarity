import sys
from unittest.mock import patch

from app.models import SimilarityMetric
from app.services.similarity_service import TextSimilarityService


class TestTextSimilarityService:
    def setup_method(self):
        self.service = TextSimilarityService()

    def test_cosine_similarity_with_identical_texts(self):
        text = "This is a test sentence."
        similarity = self.service.cosine_similarity_tfidf(text, text)
        assert similarity == 1.0

    def test_cosine_similarity_with_different_texts(self):
        text1 = "This is a cats."
        text2 = "This is a dog."
        similarity = self.service.cosine_similarity_tfidf(text1, text2)
        assert 0.0 <= similarity <= 1.0

    def test_jaccard_similarity_with_identical_texts(self):
        text = "bird cat dog"
        similarity = self.service.jaccard_similarity(text, text)
        assert similarity == 1.0

    def test_jaccard_similarity_with_no_overlap(self):
        text1 = "cat dog"
        text2 = "bird fish"
        similarity = self.service.jaccard_similarity(text1, text2)
        assert similarity == 0.0

    def test_jaccard_similarity_with_partial_overlap(self):
        text1 = "cat dog bird"
        text2 = "cat fish horse"
        similarity = self.service.jaccard_similarity(text1, text2)
        expected = 1 / 5  # 1 common word, 5 total unique words
        assert abs(similarity - expected) <= sys.float_info.epsilon

    def test_semantic_similarity_with_similar_meaning(self):
        text1 = "I'm happy"
        text2 = "I'm full of happiness"
        similarity = self.service.semantic_similarity(text1, text2)
        assert similarity > 0.5  # Expect high similarity for semantically similar texts

    def test_calculate_similarity_with_all_metrics(self):
        text1 = "This is a test"
        text2 = "This is another test"
        for metric in SimilarityMetric:
            similarity = self.service.calculate_similarity(text1, text2, metric)
            assert 0.0 <= similarity <= 1.0

    # Error condition tests

    def test_empty_text_inputs(self):
        """Test handling of empty text inputs."""
        # Empty strings should not crash
        similarity = self.service.jaccard_similarity("", "")
        assert similarity == 0.0

        similarity = self.service.jaccard_similarity("hello", "")
        assert similarity == 0.0

    def test_none_text_inputs(self):
        """Test handling of None text inputs."""
        # The service catches exceptions and returns 0.0
        similarity = self.service.jaccard_similarity(None, "hello")
        assert similarity == 0.0

    def test_whitespace_only_texts(self):
        """Test handling of whitespace-only texts."""
        similarity = self.service.jaccard_similarity("   ", "   ")
        assert similarity == 0.0  # No words after splitting

        similarity = self.service.jaccard_similarity("hello", "   ")
        assert similarity == 0.0

    @patch('app.services.similarity_service.SentenceTransformer')
    def test_semantic_model_loading_failure(self, mock_transformer):
        """Test when semantic model fails to load."""
        mock_transformer.side_effect = Exception("Model loading failed")

        service = TextSimilarityService()
        assert service.semantic_model is None

        # Should fall back to cosine similarity
        similarity = service.semantic_similarity("hello", "world")
        assert 0.0 <= similarity <= 1.0

    def test_tfidf_exception_handling(self):
        """Test TF-IDF calculation exception handling."""
        with patch('app.services.similarity_service.TfidfVectorizer') as mock_vectorizer:
            mock_vectorizer.side_effect = Exception("Vectorizer failed")

            similarity = self.service.cosine_similarity_tfidf("hello", "world")
            assert similarity == 0.0  # Should return default value
