"""
REST endpoints — /api/v1/research and /api/v1/topic.

Request bodies deliberately mirror what the Athene AI frontend already
sends: multipart form fields for the multimodal run, plain JSON for the
topic lookup.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.models.domain import GraphState, UploadedFileRef
from app.models.request import TopicAnalysisRequest
from app.models.response import HealthResponse, ResearchResponse, TopicAnalysisResponse
from app.services.orchestrator import ResearchOrchestrator, TopicAnalysisPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["research"])

# Instantiated once per process — building the LangGraph graph and the LLM
# clients is not free, so we don't want to redo it on every request.
_orchestrator: Optional[ResearchOrchestrator] = None
_topic_pipeline: Optional[TopicAnalysisPipeline] = None


def get_orchestrator() -> ResearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ResearchOrchestrator()
    return _orchestrator


def get_topic_pipeline() -> TopicAnalysisPipeline:
    global _topic_pipeline
    if _topic_pipeline is None:
        _topic_pipeline = TopicAnalysisPipeline()
    return _topic_pipeline


_MODALITY_BY_FIELD = {
    "image": "image",
    "video": "video",
    "pdf": "pdf",
    "code": "code",
}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/research", response_model=ResearchResponse, response_model_by_alias=True)
async def run_research(
    prompt: Optional[str] = Form(default=None),
    url: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    pdf: Optional[UploadFile] = File(default=None),
    code: Optional[UploadFile] = File(default=None),
) -> ResearchResponse:
    """
    Runs the full 7-agent pipeline (Planner included, per the backend
    architecture — it's simply not surfaced as its own stage in the UI).
    """
    settings = get_settings()
    uploads = {"image": image, "video": video, "pdf": pdf, "code": code}

    if not prompt and not url and not any(uploads.values()):
        raise HTTPException(status_code=400, detail="Provide at least a prompt, a URL, or one file.")

    file_refs: list[UploadedFileRef] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for field_name, upload in uploads.items():
            if upload is None:
                continue
            file_refs.append(await _persist_upload(upload, field_name, tmp_dir, settings.max_upload_mb))

        orchestrator = get_orchestrator()
        final_state: GraphState = await orchestrator.run(prompt=prompt, url=url, files=file_refs)

    return _to_research_response(final_state)


@router.post("/topic", response_model=TopicAnalysisResponse, response_model_by_alias=True)
async def run_topic_analysis(payload: TopicAnalysisRequest) -> TopicAnalysisResponse:
    """Powers the frontend's standalone 'Research Analysis' section."""
    pipeline = get_topic_pipeline()
    result = await pipeline.run(payload.topic)
    return TopicAnalysisResponse(
        overview=result.overview,
        origin=result.origin,
        keyFacts=result.key_facts,
        credibility=result.credibility,
        triples=[t.as_tuple() for t in result.triples],
    )


async def _persist_upload(upload: UploadFile, field_name: str, tmp_dir: str, max_upload_mb: int) -> UploadedFileRef:
    dest_path = Path(tmp_dir) / upload.filename

    def _write() -> int:
        size = 0
        with dest_path.open("wb") as out_file:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_upload_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds {max_upload_mb}MB limit")
                out_file.write(chunk)
        return size

    size_bytes = await run_in_threadpool(_write)
    logger.info("Persisted upload %s (%d bytes) as modality=%s", upload.filename, size_bytes, field_name)

    return UploadedFileRef(
        modality=_MODALITY_BY_FIELD[field_name],
        filename=upload.filename or field_name,
        content_type=upload.content_type or "application/octet-stream",
        size_bytes=size_bytes,
        storage_path=str(dest_path),
    )


def _to_research_response(state: GraphState) -> ResearchResponse:
    return ResearchResponse(
        report=state.get("report", ""),
        confidence=state.get("confidence_score", 0.0),
        inputScore=state.get("input_score", 0.0),
        recommendedScore=state.get("recommended_score", 0.0),
        recommendations=state.get("recommendations", []),
        triples=state.get("triples", []),
        sources=state.get("sources", []),
    )
