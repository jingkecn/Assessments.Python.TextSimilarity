from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service configuration
    SERVICE_NAME: str = "text-similarity-service"
    VERSION: str = "1.0.0"


settings = Settings()
