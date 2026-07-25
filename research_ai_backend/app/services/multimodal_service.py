"""
Multimodal parsing service.

Pure I/O + parsing logic lives here, kept separate from `ingestion_agent.py`
so the agent stays a thin orchestration layer while this module owns the
actual PyMuPDF / OCR / OpenCV calls. Each parser is defensive: a failure in
one modality (e.g. a corrupt PDF) never raises past this module — it logs
and returns an empty string so the rest of the pipeline can continue with
whatever other modalities were provided.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def parse_pdf(path: str, max_chars: int = 12000) -> str:
    """Extract text (and falls back to per-page OCR for scanned PDFs) via PyMuPDF."""
    def _extract() -> str:
        import fitz  # PyMuPDF

        text_parts: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text().strip()
                if page_text:
                    text_parts.append(page_text)
                else:
                    # No extractable text layer -> likely a scanned page; OCR it.
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    text_parts.append(_ocr_image_bytes(img_bytes))
        return "\n".join(text_parts)[:max_chars]

    try:
        return await asyncio.to_thread(_extract)
    except Exception:
        logger.exception("Failed to parse PDF at %s", path)
        return ""


def _ocr_image_bytes(img_bytes: bytes) -> str:
    """Run EasyOCR on raw image bytes. Isolated so it can be swapped for Tesseract."""
    try:
        import io

        import easyocr
        import numpy as np
        from PIL import Image

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = reader.readtext(np.array(image), detail=0)
        return " ".join(result)
    except Exception:
        logger.exception("OCR failed for an image/scanned page")
        return ""


async def parse_image(path: str) -> str:
    """OCR + lightweight description for a standalone image upload."""
    def _read() -> str:
        return _ocr_image_bytes(Path(path).read_bytes())

    try:
        text = await asyncio.to_thread(_read)
        return text or "[Image contains no machine-readable text]"
    except Exception:
        logger.exception("Failed to parse image at %s", path)
        return ""


async def parse_video(path: str, sample_every_n_seconds: int = 5, max_frames: int = 6) -> dict:
    """
    Extract a handful of representative frames (OpenCV) and an audio
    transcript (Whisper) from a video file.

    Returns {"frame_captions": [...], "transcript": "..."}.
    """
    def _sample_frames() -> list[str]:
        import cv2

        captions: list[str] = []
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = max(int(fps * sample_every_n_seconds), 1)
        frame_idx = 0
        while len(captions) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                ok2, buf = cv2.imencode(".png", frame)
                if ok2:
                    captions.append(_ocr_image_bytes(buf.tobytes()))
            frame_idx += 1
        cap.release()
        return [c for c in captions if c.strip()]

    try:
        frame_captions = await asyncio.to_thread(_sample_frames)
    except Exception:
        logger.exception("Failed to sample frames from %s", path)
        frame_captions = []

    transcript = await transcribe_audio(path)
    return {"frame_captions": frame_captions, "transcript": transcript}


async def transcribe_audio(path: str) -> str:
    """
    Transcribe the audio track of a file with Whisper.

    Loading the Whisper model is expensive, so in a real deployment this
    should be a singleton loaded once at startup (see `main.py`'s lifespan
    hook) rather than reloaded per request — kept simple here for clarity.
    """
    def _transcribe() -> str:
        import whisper

        from app.config import get_settings

        model = whisper.load_model(get_settings().whisper_model_size)
        result = model.transcribe(path, fp16=False)
        return result.get("text", "").strip()

    try:
        return await asyncio.to_thread(_transcribe)
    except Exception:
        logger.exception("Transcription failed for %s", path)
        return ""


async def parse_code_file(path: str) -> str:
    """Read a code file as plain text (execution, if any, is opt-in via code_sandbox)."""
    try:
        return await asyncio.to_thread(lambda: Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        logger.exception("Failed to read code file at %s", path)
        return ""
