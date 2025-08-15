from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.models import SimilarityResponse, SimilarityRequest, HealthResponse, SimilarityMetric
from app.services.similarity_service import TextSimilarityService
from app.utils.config import settings

# Global service instances
similarity_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown."""
    global similarity_service

    print("Starting up text similarity service...")

    # Initialize services
    similarity_service = TextSimilarityService()

    print("Service initialization complete")

    yield

    print("Shutting down text similarity service...")


# Create FastAPI app
app = FastAPI(
    title="Text Similarity Service",
    description="A simple service for computing text similarity",
    version="1.0.0",
    lifespan=lifespan
)


# Dependency injection for services
def get_similarity_service() -> TextSimilarityService:
    if similarity_service is None:
        raise HTTPException(status_code=503, detail="Similarity service not initialized")
    return similarity_service


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        is_llm_available=False,
        service=settings.SERVICE_NAME,
        status="healthy",
        version=settings.VERSION
    )


@app.get("/metrics")
async def get_available_metrics():
    """Get list of available similarity metrics."""
    return {
        "available_metrics": [m.value for m in SimilarityMetric],
        "descriptions": {
            "cosine": "Cosine similarity using TF-IDF vectors",
            "jaccard": "Jaccard similarity based on word overlap",
            "semantic": "Semantic similarity using sentence transformers"
        }
    }


@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(
        request: SimilarityRequest,
        similarity_svc: TextSimilarityService = Depends(get_similarity_service)
) -> SimilarityResponse:
    """
    Calculate text similarity between two prompts.

    This endpoint:
    1. Sanitizes input prompts (TODO)
    2. Calculates similarity using specified metric
    3. If prompts are similar enough, sends one to LLM (TODO)
    4. Sanitized and returns the response (TODO)
    """
    try:
        similarity_score = similarity_svc.calculate_similarity(
            request.prompt1,
            request.prompt2,
            request.similarity_metric
        )

        success = similarity_score >= request.similarity_threshold

        response = SimilarityResponse(
            are_similar=success,
            llm_response="The prompts are similar! Here's a hardcoded response for demonstration." if success else None,
            similarity_metric=request.similarity_metric,
            similarity_score=similarity_score
        )

        return response
    except HTTPException:
        raise
    # Let ValueError and other exceptions propagate to global handlers


# Error handlers

@app.exception_handler(ValueError)
async def handle_value_error(request, exception: Exception) -> JSONResponse:
    print(f"Value error in request, error={exception}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exception)}
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, exc: RequestValidationError) -> JSONResponse:
    print(f"Validation error in request: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "detail": exc.errors()}
    )


@app.exception_handler(500)
async def handle_internal_error(request, exception: Exception) -> JSONResponse:
    print(f"Internal server error, error={exception}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=44101)
