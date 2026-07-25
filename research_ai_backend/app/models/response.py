"""
Response schemas.

These shapes intentionally mirror what the Athene AI frontend already
expects to parse (see `renderResults()` / `renderTopicResult()` in the
frontend's JS) so the two can be wired together with no translation layer.
"""
from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

from app.models.domain import Recommendation, SourceRef


class ResearchResponse(BaseModel):
    """Response for POST /api/v1/research — the full multimodal pipeline."""
    report: str
    confidence: float = Field(ge=0, le=100)
    input_score: float = Field(ge=0, le=100, alias="inputScore")
    recommended_score: float = Field(ge=0, le=100, alias="recommendedScore")
    recommendations: List[Recommendation] = Field(default_factory=list)
    triples: List[Tuple[str, str, str]] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class TopicAnalysisResponse(BaseModel):
    """Response for POST /api/v1/topic — the standalone Research Analysis lookup."""
    overview: str
    origin: str
    key_facts: List[str] = Field(default_factory=list, alias="keyFacts")
    credibility: float = Field(ge=0, le=100)
    triples: List[Tuple[str, str, str]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str = "ok"
    orchestrator: str = "online"
