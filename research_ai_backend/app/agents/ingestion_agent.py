"""
STAGE 1 — Multimodal Ingestion Agent
Pre-processing & context standardizer.

Turns whatever combination of image / video / PDF / code / URL / prompt the
user attached into a single `standardized_text` string that every
downstream agent can reason over, without needing to know anything about
the original modalities.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.domain import GraphState, UploadedFileRef
from app.services import multimodal_service
from app.tools.code_sandbox import summarize_code_for_ingestion
from app.tools.web_scraper import fetch_page_text

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Stage 1 node: parses every attached modality into unified text."""

    name = "ingestion"

    async def run(self, state: GraphState) -> dict[str, Any]:
        segments: list[str] = []
        modalities_used: list[str] = []

        prompt = state.get("raw_prompt")
        if prompt:
            segments.append(f"[User prompt]\n{prompt}")
            modalities_used.append("prompt")

        url = state.get("raw_url")
        if url:
            page_text = await fetch_page_text(url)
            if page_text:
                segments.append(f"[Web URL: {url}]\n{page_text}")
                modalities_used.append("url")

        for file_ref in state.get("raw_files", []):
            text = await self._parse_file(file_ref)
            if text:
                segments.append(f"[{file_ref.modality.upper()}: {file_ref.filename}]\n{text}")
                modalities_used.append(file_ref.modality)

        standardized_text = "\n\n".join(segments).strip()
        if not standardized_text:
            standardized_text = "[No content could be extracted from the provided inputs.]"

        logger.info("Ingestion complete — modalities used: %s", modalities_used)
        return {
            "standardized_text": standardized_text,
            "modalities_used": modalities_used,
        }

    async def _parse_file(self, file_ref: UploadedFileRef) -> str:
        try:
            if file_ref.modality == "pdf":
                return await multimodal_service.parse_pdf(file_ref.storage_path)
            if file_ref.modality == "image":
                return await multimodal_service.parse_image(file_ref.storage_path)
            if file_ref.modality == "video":
                result = await multimodal_service.parse_video(file_ref.storage_path)
                parts = []
                if result["transcript"]:
                    parts.append(f"Transcript: {result['transcript']}")
                if result["frame_captions"]:
                    parts.append("On-screen text: " + " | ".join(result["frame_captions"]))
                return "\n".join(parts)
            if file_ref.modality == "code":
                source = await multimodal_service.parse_code_file(file_ref.storage_path)
                return summarize_code_for_ingestion(source, file_ref.filename)
        except Exception:
            logger.exception("Ingestion failed for %s (%s)", file_ref.filename, file_ref.modality)
        return ""
