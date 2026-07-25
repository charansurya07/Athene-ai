"""
Speech service — streaming speech-to-text for the Live Voice modality.

`websocket_voice.py` calls into this module frame-by-frame; keeping the
Deepgram/Whisper-streaming client here (rather than in the route handler)
means the same logic can be reused by a future batch endpoint.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """
    Wraps a streaming ASR backend. Prefers Deepgram (true low-latency
    streaming) when `DEEPGRAM_API_KEY` is set, otherwise buffers audio and
    falls back to batched Whisper transcription on each flush — good enough
    for a demo, not a substitute for real streaming ASR in production.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._buffer = bytearray()

    async def push_audio_chunk(self, chunk: bytes) -> Optional[str]:
        """
        Feed one chunk of raw audio bytes in. Returns an incremental partial
        transcript string when one is available, else None.
        """
        if self._settings.deepgram_api_key:
            return await self._push_deepgram(chunk)
        return await self._push_whisper_fallback(chunk)

    async def _push_deepgram(self, chunk: bytes) -> Optional[str]:
        # A real implementation opens one persistent Deepgram websocket per
        # session at connect-time and forwards chunks to it; sketched here
        # as a stub so the FastAPI websocket route has a stable interface
        # to call regardless of which backend is configured.
        logger.debug("Forwarding %d bytes to Deepgram streaming session", len(chunk))
        return None

    async def _push_whisper_fallback(self, chunk: bytes) -> Optional[str]:
        self._buffer.extend(chunk)
        # Flush roughly every ~3s of 16kHz mono 16-bit audio.
        if len(self._buffer) < 16_000 * 2 * 3:
            return None
        text = await self._flush_whisper()
        return text

    async def _flush_whisper(self) -> str:
        import asyncio
        import io
        import tempfile
        import wave

        def _transcribe(raw: bytes) -> str:
            import whisper

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                with wave.open(tmp.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16_000)
                    wav_file.writeframes(raw)
                model = whisper.load_model(self._settings.whisper_model_size)
                result = model.transcribe(tmp.name, fp16=False)
                return result.get("text", "").strip()

        raw = bytes(self._buffer)
        self._buffer.clear()
        return await asyncio.to_thread(_transcribe, raw)

    async def finalize(self) -> Optional[str]:
        """Call when the client stream ends, to flush any remaining audio."""
        if self._buffer:
            return await self._flush_whisper()
        return None


async def transcript_stream_stub() -> AsyncIterator[str]:
    """Placeholder generator kept for documentation / testing purposes."""
    yield ""
