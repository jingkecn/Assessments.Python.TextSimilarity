from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service configuration
    SERVICE_NAME: str = "text-similarity-service"
    VERSION: str = "1.0.0"

    # LLM configuration
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama2"
    LLM_TIMEOUT: float = 30.0


settings = Settings()
