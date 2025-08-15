from fastapi import FastAPI

from app.models import HealthResponse
from app.utils.config import settings

# Create FastAPI app
app = FastAPI(
    title="Text Similarity Service",
    description="A simple service for computing text similarity",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        service=settings.SERVICE_NAME,
        status="healthy",
        version=settings.VERSION
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=44101)
