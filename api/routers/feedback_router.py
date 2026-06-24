"""Routes for audio feedback generation."""
import asyncio
import datetime
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.core.constants import DEFAULT_FEEDBACK_TEXT
from api.services.feedback_service import generate_feedback_text
from api.services.text_analysis_service import analyse_filler_words
from api.services.transcription_service import transcribe_audio
from api.services.tts_service import TTSError, synthesise

feedback_router = APIRouter()


class FeedbackTextBody(BaseModel):
    """Request body for feedback text generation."""

    analysis: dict | None = None
    category: str | None = None


@feedback_router.post("/api/audio-feedback")
async def generate_audio_feedback(
    file: UploadFile | None = File(None),
    category: str | None = Form(None),
) -> StreamingResponse:
    """Generate spoken feedback from an uploaded audio file.

    Args:
        file: Audio file to analyse (optional)
        category: Practice category (optional)

    Returns:
        WAV audio stream with analysis in X-Speech-Metrics header

    Raises:
        HTTPException: If TTS generation or processing fails
    """
    try:
        analysis: dict | None = None
        speech_speed = 1.0

        if file and file.filename:
            temp_input = await _save_upload(file)
            try:
                loop = asyncio.get_event_loop()
                try:
                    analysis = await asyncio.wait_for(
                        loop.run_in_executor(None, _process_audio_file, temp_input),
                        timeout=120,
                    )
                except asyncio.TimeoutError:
                    logging.error("Audio feedback processing timed out after 120s")
                    analysis = None
            finally:
                temp_input.unlink(missing_ok=True)

        feedback_text = generate_feedback_text(
            analysis=analysis,
            practice_category=category,
            default_text=DEFAULT_FEEDBACK_TEXT,
        )

        logging.info("Generating TTS for feedback: %d chars", len(feedback_text))
        buffer = synthesise(feedback_text, speed=speech_speed)

        metrics_dict: dict[str, Any] = analysis if isinstance(analysis, dict) else {}
        if feedback_text:
            metrics_dict["feedback_text"] = feedback_text

        headers = {
            "Content-Disposition": "attachment; filename=feedback_speech.wav",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, "
                "X-Speech-Metrics, X-Practice-Category"
            ),
            "X-Speech-Metrics": json.dumps(metrics_dict),
            "X-Practice-Category": category or "unknown",
        }

        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    except TTSError as e:
        logging.error("TTS error in audio feedback: %s", e)
        raise HTTPException(status_code=e.status_code, detail=str(e))

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Feedback generation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _save_upload(file: UploadFile) -> Path:
    """Save an uploaded file to a unique temp path.

    Args:
        file: Incoming upload

    Returns:
        Path to the saved temp file
    """
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    unique_id = (
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + "_" + str(uuid.uuid4())[:8]
    )
    temp_input = temp_dir / f"input_{unique_id}.wav"
    temp_input.write_bytes(await file.read())
    return temp_input


def _process_audio_file(temp_input: Path) -> dict | None:
    """Transcribe and analyse an audio file.

    Args:
        temp_input: Path to the audio file on disk

    Returns:
        Analysis dict from analyse_filler_words, or None on failure
    """
    if not temp_input.stat().st_size:
        return None

    temp_audio = temp_input.with_suffix(".mp3")

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(temp_input), "-vn",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
            "-b:a", "192k",
            "-af", "highpass=f=50,lowpass=f=15000,volume=2,afftdn=nf=-20",
            str(temp_audio),
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)

        if not (temp_audio.exists() and temp_audio.stat().st_size):
            return None

        text = transcribe_audio(str(temp_audio))
        if not text:
            return None

        logging.info("Analysing transcribed text: %d chars", len(text))
        return analyse_filler_words(text)

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Audio processing failures are non-fatal; fallback to default feedback
        logging.error("Audio processing error: %s", e)
        return None

    finally:
        temp_audio.unlink(missing_ok=True)


@feedback_router.post("/api/generate-feedback-text")
def generate_feedback_text_route(body: FeedbackTextBody) -> dict[str, Any]:
    """Generate feedback text without TTS for frontend caching.

    Args:
        body: Analysis dict and practice category

    Returns:
        Dict with feedback_text and category fields

    Raises:
        HTTPException: If generation fails
    """
    try:
        feedback_text = generate_feedback_text(
            analysis=body.analysis,
            practice_category=body.category,
            default_text=DEFAULT_FEEDBACK_TEXT,
        )

        logging.info("Generated feedback text: %d chars", len(feedback_text))
        return {"feedback_text": feedback_text, "category": body.category}

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Feedback text generation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
