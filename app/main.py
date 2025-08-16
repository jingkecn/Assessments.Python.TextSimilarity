from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from app.models import SimilarityResponse, SimilarityRequest, HealthResponse, SimilarityMetric
from app.services.cache_service import CacheService
from app.services.llm_service import LLMService
from app.services.sanitization_service import TextSanitizationService
from app.services.similarity_service import TextSimilarityService
from app.utils.config import settings

# Global service instances
cache_service = None
llm_service = None
sanitization_service = None
similarity_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown."""
    global cache_service, llm_service, sanitization_service, similarity_service

    print("Starting up text similarity service...")

    # Initialize services
    cache_service = CacheService()
    llm_service = LLMService(
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        temperature=settings.LLM_TEMPERATURE
    )
    sanitization_service = TextSanitizationService()
    similarity_service = TextSimilarityService(cache_service)

    # Check LLM availability
    _ = await llm_service.is_available()

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
def get_cache_service() -> CacheService:
    if cache_service is None:
        raise HTTPException(status_code=503, detail="Cache service not initialized")
    return cache_service


def get_similarity_service() -> TextSimilarityService:
    if similarity_service is None:
        raise HTTPException(status_code=503, detail="Similarity service not initialized")
    return similarity_service


def get_sanitization_service() -> TextSanitizationService:
    if sanitization_service is None:
        raise HTTPException(status_code=503, detail="Sanitization service not initialized")
    return sanitization_service


def get_llm_service() -> LLMService:
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service not initialized")
    return llm_service


@app.get("/health", response_model=HealthResponse)
async def health_check(
        llm_svc: LLMService = Depends(get_llm_service)
) -> HealthResponse:
    """Health check endpoint."""
    is_llm_available = await llm_svc.is_available()
    return HealthResponse(
        environment=settings.ENVIRONMENT,
        is_llm_available=is_llm_available,
        llm_model=settings.LLM_MODEL,
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
        llm_svc: LLMService = Depends(get_llm_service),
        sanitization_svc: TextSanitizationService = Depends(get_sanitization_service),
        similarity_svc: TextSimilarityService = Depends(get_similarity_service)
) -> SimilarityResponse:
    """
    Calculate text similarity between two prompts.

    This endpoint:
    1. Sanitizes input prompts
    2. Calculates similarity using specified metric
    3. If prompts are similar enough, sends one to LLM
    4. Sanitized and returns the response
    """
    try:
        prompt1 = sanitization_svc.sanitize_text(request.prompt1)
        prompt2 = sanitization_svc.sanitize_text(request.prompt2)
        if prompt1 != request.prompt1 or prompt2 != request.prompt2:
            raise ValueError(f"Input sanitized: prompt1='{prompt1}', prompt2='{prompt2}'")

        similarity_score = similarity_svc.calculate_similarity(
            prompt1,
            prompt2,
            request.similarity_metric
        )
        success = similarity_score >= request.similarity_threshold
        response = SimilarityResponse(
            are_similar=success,
            similarity_metric=request.similarity_metric,
            similarity_score=similarity_score
        )

        if not request.use_llm or not success:
            return response

        llm_response = await llm_svc.generate_response_with_retry(request.prompt1)
        if not llm_response:
            print("LLM failed to generate response")
            llm_response = "LLM service unavailable or failed to generate response"

        response.llm_response = sanitization_svc.sanitize_text(llm_response)
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

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
