"""
Domain models shared by every agent.

`GraphState` is the single object that flows through the LangGraph graph —
each agent node reads the fields it needs and returns a partial dict that
LangGraph merges back into the state. Keeping it as one TypedDict (rather
than passing bespoke arguments between agents) is what lets the 7 agents
stay decoupled: every agent only needs to import `GraphState`.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, TypedDict

from pydantic import BaseModel, Field

Modality = Literal["image", "video", "pdf", "code", "url", "prompt", "voice", "sql"]


class UploadedFileRef(BaseModel):
    """Reference to a single uploaded file, before it has been parsed."""
    modality: Modality
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str  # where multimodal_service wrote the temp copy


class SourceRef(BaseModel):
    title: str
    url: str
    credibility: float = Field(ge=0, le=100)
    published_at: Optional[str] = None
    snippet: Optional[str] = None


class Recommendation(BaseModel):
    claim: str
    issue: str
    alternative: str


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class GraphState(TypedDict, total=False):
    """State object threaded through every LangGraph node."""

    request_id: str

    # ---- raw input (populated by the API layer before Stage 1) ----
    raw_prompt: str
    raw_url: Optional[str]
    raw_files: List[UploadedFileRef]

    # ---- Stage 1: Ingestion Agent ----
    standardized_text: str
    modalities_used: List[Modality]

    # ---- Stage 2: Planner Agent ----
    sub_queries: List[str]

    # ---- Stage 3: Searcher Agent ----
    search_results: List[dict[str, Any]]

    # ---- Stage 4: Verifier Agent ----
    verified_facts: List[dict[str, Any]]
    confidence_score: float
    sources: List[SourceRef]

    # ---- Stage 5: Recommendation Agent ----
    recommendations: List[Recommendation]
    input_score: float
    recommended_score: float

    # ---- Stage 6: Knowledge Graph Agent ----
    triples: List[Tuple[str, str, str]]

    # ---- Stage 7: Writer Agent ----
    report: str

    # ---- bookkeeping ----
    errors: List[str]
