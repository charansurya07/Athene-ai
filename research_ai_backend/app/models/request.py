"""Request schemas accepted by the FastAPI endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ResearchRequestMeta(BaseModel):
    """
    Non-file fields of a multimodal research request.

    Files themselves arrive as `UploadFile`s alongside this metadata in a
    multipart/form-data body — see `app/api/endpoints.py`.
    """
    prompt: Optional[str] = Field(default=None, description="Free-text research prompt / question")
    url: Optional[str] = Field(default=None, description="A web URL to ingest and analyze")
    session_id: Optional[str] = Field(default=None, description="Client-generated session/user id")


class TopicAnalysisRequest(BaseModel):
    """Payload for the lightweight 'Research Analysis' (topic lookup) endpoint."""
    topic: str = Field(..., min_length=1, max_length=300)
