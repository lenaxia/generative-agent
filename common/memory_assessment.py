"""Memory assessment models for LLM-based importance scoring.

This module provides Pydantic models for structured memory assessment output
from LLM calls. The assessment includes importance scoring, summary generation,
tag extraction, and topic identification.
"""

from pydantic import BaseModel, Field


class MemoryAssessment(BaseModel):
    """Pydantic model for memory importance assessment.

    This model defines the structured output expected from the LLM when
    assessing the importance of a memory. It includes validation rules
    to ensure the LLM output is well-formed.

    Attributes:
        importance: Importance score between 0.0 and 1.0
        summary: Concise summary of the interaction
        tags: 1-10 lowercase tags for categorization
        topics: 0-5 main topics discussed
        reasoning: Brief explanation of importance score (max 500 chars)
    """

    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance score between 0.0 and 1.0",
    )
    summary: str = Field(
        ..., min_length=10, description="Concise summary of the interaction"
    )
    tags: list[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Lowercase tags for categorization",
    )
    topics: list[str] = Field(
        default_factory=list,
        max_items=5,
        description="Main topics discussed (0-5)",
    )
    reasoning: str = Field(
        ..., max_length=500, description="Brief explanation of importance score"
    )

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"  # Don't allow extra fields
