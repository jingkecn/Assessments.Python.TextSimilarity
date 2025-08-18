import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = os.environ.get("API_PORT", 44101)

    # Environment
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", 'development')

    # LLM configuration
    LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama2")
    LLM_TIMEOUT: float = os.environ.get("LLM_TIMEOUT", 30.0)
    LLM_MAX_RETRIES: int = os.environ.get("LLM_MAX_RETRIES", 3)
    LLM_TEMPERATURE: float = os.environ.get("LLM_TEMPERATURE", 0.7)

    # Logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # Service configuration
    SERVICE_NAME: str = os.environ.get('SERVICE_NAME', "text-similarity-service")
    VERSION: str = os.environ.get('VERSION', "1.0.0")

    # Sentence Transformer Model
    SENTENCE_TRANSFORMER_MODEL: str = os.environ.get('SENTENCE_TRANSFORMER_MODEL', "all-MiniLM-L6-v2")

    # Worker configuration
    WEB_CONCURRENCY: int = int(os.environ.get("WEB_CONCURRENCY", os.cpu_count()))


settings = Settings()
