"""Miscellaneous API routes for health checks and text-to-speech."""
import logging
from typing import Any

import dlib
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.services.tts_service import TTSError, synthesise

misc_router = APIRouter()


class TTSBody(BaseModel):
    """Request body for TTS generation."""

    text: str
    voice: str | None = None
    speed: float = 1.05
    category: str | None = None


@misc_router.get("/api/test")
def test_endpoint() -> dict[str, Any]:
    """Verify core API functionality.

    Returns:
        Status dict with component availability flags
    """
    try:
        dlib.get_frontal_face_detector()

        emotion_model_available = False
        try:
            from deepface import DeepFace  # pylint: disable=import-outside-toplevel
            _ = DeepFace
            emotion_model_available = True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Emotion detection unavailable: %s", e)

        return {
            "status": "ok",
            "message": "Backend API is running",
            "details": {
                "face_detection": True,
                "emotion_model": emotion_model_available,
                "version": "1.0.0",
            },
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Health check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@misc_router.post("/api/tts-core")
def tts_core_endpoint(body: TTSBody) -> StreamingResponse:
    """Generate speech audio from text.

    Args:
        body: TTS request with text, voice, speed, and optional category

    Returns:
        WAV audio stream

    Raises:
        HTTPException: If TTS generation fails
    """
    try:
        logging.info("TTS request: %d chars", len(body.text))
        buffer = synthesise(body.text, body.voice, body.speed)

        headers = {
            "Content-Disposition": "attachment; filename=tts_speech.wav",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, X-Practice-Category"
            ),
        }
        if body.category:
            headers["X-Practice-Category"] = body.category

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers=headers,
        )

    except TTSError as e:
        logging.error("TTS error: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("TTS endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
