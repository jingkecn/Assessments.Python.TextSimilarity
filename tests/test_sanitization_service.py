from app.services.sanitization_service import TextSanitizationService


class TestTextSanitizationService:
    def setup_method(self):
        self.service = TextSanitizationService()

    def test_profanity_censor(self):
        text = "This is a damn test."
        sanitized = self.service.sanitize_text(text)
        assert "****" in sanitized or "*" in sanitized  # better_profanity replaces with asterisks

    def test_disallowed_phrase(self):
        text = "I want to hack into systems."
        sanitized = self.service.sanitize_text(text)
        assert "[*FORBIDDEN*]" in sanitized

    def test_harmful_pattern_script(self):
        text = "<script>alert('hack');</script>"
        sanitized = self.service.sanitize_text(text)
        assert "[*HARMFUL*]" in sanitized

    def test_harmful_pattern_javascript(self):
        text = "Click here: javascript:alert('hi')"
        sanitized = self.service.sanitize_text(text)
        assert "[*HARMFUL*]" in sanitized

    def test_sensitive_email(self):
        text = "Contact me at test@example.com."
        sanitized = self.service.sanitize_text(text)
        assert "[*SENSITIVE*]" in sanitized

    def test_sensitive_ssn_in_france(self):
        text = "My SSN is 1 23 45 67 890 123 45."
        sanitized = self.service.sanitize_text(text)
        assert "[*SENSITIVE*]" in sanitized
