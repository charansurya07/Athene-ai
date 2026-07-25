"""
WebSocket handshake for Live Voice ingestion.

Protocol: client connects to `/ws/voice`, then streams raw 16kHz mono PCM
audio chunks as binary frames. The server pushes back a JSON text frame
`{"type": "partial", "text": "..."}` whenever a partial transcript is
available, and `{"type": "final", "text": "..."}` once the client sends a
`{"type": "end"}` control message (a JSON text frame) or disconnects.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.speech_service import StreamingTranscriber

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    transcriber = StreamingTranscriber()
    logger.info("Voice websocket connected")

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes") is not None:
                partial = await transcriber.push_audio_chunk(message["bytes"])
                if partial:
                    await websocket.send_text(json.dumps({"type": "partial", "text": partial}))

            elif message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    control = {}
                if control.get("type") == "end":
                    final_text = await transcriber.finalize()
                    await websocket.send_text(json.dumps({"type": "final", "text": final_text or ""}))
                    break

    except WebSocketDisconnect:
        logger.info("Voice websocket disconnected")
    except Exception:
        logger.exception("Voice websocket error")
        await websocket.close(code=1011)
