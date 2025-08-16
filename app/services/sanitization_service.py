import re

from better_profanity import profanity


class TextSanitizationService:
    def __init__(self):
        # Initialize profanity filter
        profanity.load_censor_words()

        # Define disallowed words
        self.disallowed_phases = [
            "hack into systems",
            "how to make bombs",
            "illegal activities",
            # Add more based on requirements
        ]

        # Define patters for potential harmful content
        self.harmful_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript links
            r'<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>',  # Iframe tags
            # Add more based on requirements
        ]

        # Define patterns for sensitive information
        self.sensitive_patterns = [
            r'\b\d{1}\s\d{2}\s\d{2}\s\d{2}\s\d{3}\s\d{3}\s\d{2}\b',  # SSN format (France)
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'  # Email address
            # Add more based on requirements
        ]

        # Compile regex patterns for performance
        self.harmful_regex = [re.compile(pattern) for pattern in self.harmful_patterns]
        self.sensitive_regex = [re.compile(pattern) for pattern in self.sensitive_patterns]

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing profanity, disallowed phrases, harmful content, and sensitive information.
        :param text: Input text to sanitize
        :return: Sanitized text
        """
        # Remove profanity
        text = profanity.censor(text)

        # Remove disallowed phrases
        for phrase in self.disallowed_phases:
            text = text.replace(phrase, '[*FORBIDDEN*]')

        # Remove harmful content
        for pattern in self.harmful_regex:
            text = pattern.sub('[*HARMFUL*]', text)

        # Remove sensitive information
        for pattern in self.sensitive_regex:
            text = pattern.sub('[*SENSITIVE*]', text)

        return text.strip()
