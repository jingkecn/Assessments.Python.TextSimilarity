from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    is_llm_available: bool
    service: str
    status: str
    version: str


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    JACCARD = "jaccard"
    SEMANTIC = "semantic"


class SimilarityRequest(BaseModel):
    prompt1: str = Field(..., min_length=1, description="First text prompt")
    prompt2: str = Field(..., min_length=1, description="Second text prompt")
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric to use"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to trigger LLM call"
    )

    @field_validator("prompt1", "prompt2")
    @classmethod
    def validate_prompts(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompts cannot be empty or whitespace only")
        return value


class SimilarityResponse(BaseModel):
    are_similar: bool = Field(..., description="Whether the two prompts are similar")
    llm_response: Optional[str] = Field(None, description="Response if the two prompts are similar")
    similarity_metric: SimilarityMetric = Field(..., description="Metric used for similarity calculation")
    similarity_score: float = Field(..., description="Calculated similarity score")
