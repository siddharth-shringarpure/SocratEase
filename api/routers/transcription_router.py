"""Routes for transcription-related endpoints."""
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.services.text_analysis_service import analyse_filler_words, calculate_ttr, logical_flow
from api.services.transcription_service import transcribe_audio
from api.utils.audio_utils import allowed_audio_file

transcription_router = APIRouter()


class CleanupAudioBody(BaseModel):
    """Request body for audio file cleanup."""

    audioFilename: str


class CleanupVideoBody(BaseModel):
    """Request body for video file cleanup."""

    videoFilename: str


class TranscribeBody(BaseModel):
    """JSON body for filename-based transcription requests."""

    audioFilename: str | None = None


@transcription_router.post("/api/speech2text")
async def transcribe_request(
    request: Request,
    file: UploadFile | None = File(None),
    audio_filename: str | None = Form(None),
) -> dict[str, Any]:
    """Transcribe audio and perform speech analysis.

    Accepts either a file upload or a filename referencing a server-side file.

    Args:
        file: Audio file upload (optional)
        audio_filename: Filename of a pre-uploaded audio file (optional)

    Returns:
        Dict with success flag, transcription text, and analysis metrics

    Raises:
        HTTPException: If no input provided or transcription fails
    """
    temp_path: Path | None = None
    owns_temp = False

    try:
        logging.info("Starting transcription request")

        # Accept JSON body when no multipart file is present
        if not file and not audio_filename:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    body = await request.json()
                    audio_filename = body.get("audioFilename") or body.get("audio_filename")
                except Exception:
                    pass

        if file and file.filename:
            if not allowed_audio_file(file.filename):
                raise HTTPException(status_code=400, detail="Invalid file type")

            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            safe_name = Path(file.filename).name
            temp_path = temp_dir / safe_name
            temp_path.write_bytes(await file.read())
            owns_temp = True
            logging.info("Received file: %s", file.filename)

        elif audio_filename:
            base_filename = (
                audio_filename if audio_filename.endswith(".wav")
                else f"{audio_filename}.wav"
            )
            candidates = [
                Path(os.getcwd()) / "uploads" / base_filename,
                Path(os.getcwd()) / "public" / "uploads" / base_filename,
            ]
            for candidate in candidates:
                if candidate.exists():
                    temp_path = candidate
                    break

            if not temp_path:
                raise HTTPException(status_code=404, detail="Audio file not found")

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a file upload or audio_filename",
            )

        logging.info("Processing file at: %s", temp_path)

        try:
            loop = asyncio.get_event_loop()
            transcription = await asyncio.wait_for(
                loop.run_in_executor(None, transcribe_audio, str(temp_path)),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logging.error("Transcription timed out after 120s")
            raise HTTPException(status_code=504, detail="Transcription timed out")

        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        analysis = analyse_filler_words(transcription)
        analysis["ttr_analysis"] = calculate_ttr(transcription)

        try:
            flow_result = logical_flow(transcription)
            if flow_result > 0:
                analysis["logical_flow"]["score"] = flow_result
        except Exception as flow_error:  # pylint: disable=broad-exception-caught
            logging.warning("Using fallback logical flow score: %s", flow_error)

        return {
            "success": True,
            "text": transcription,
            "analysis": analysis,
        }

    except HTTPException:
        raise

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error processing audio: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

    finally:
        if owns_temp and temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as e:
                logging.error("Temp file cleanup failed: %s", e)


@transcription_router.post("/api/cleanup-audio")
def cleanup_audio_file(body: CleanupAudioBody) -> dict[str, Any]:
    """Delete an audio file from the server.

    Args:
        body: Filename of the audio file to remove

    Returns:
        Dict with success flag and message

    Raises:
        HTTPException: If deletion fails or file not found
    """
    filename = body.audioFilename
    if not filename.endswith(".wav"):
        filename = f"{filename}.wav"

    possible_paths = [
        Path(os.getcwd()) / "uploads" / filename,
        Path(os.getcwd()) / "public" / "uploads" / filename,
        Path(os.getcwd()) / "temp" / filename,
        Path(tempfile.gettempdir()) / filename,
    ]

    logging.info("Checking for audio file %s", filename)

    for path in possible_paths:
        if path.exists():
            try:
                path.unlink()
                logging.info("✓ Deleted audio file: %s", filename)
                return {"success": True, "message": f"Deleted audio file: {filename}"}
            except OSError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Audio file deletion failed: {e}",
                )

    raise HTTPException(status_code=404, detail="Audio file not found")


@transcription_router.post("/api/cleanup-video")
def cleanup_video_file(body: CleanupVideoBody) -> dict[str, Any]:
    """Delete a video file from the server.

    Args:
        body: Filename of the video file to remove

    Returns:
        Dict with success flag and message

    Raises:
        HTTPException: If deletion fails or file not found
    """
    filename = body.videoFilename
    if not filename.endswith(".mp4"):
        filename = f"{filename}.mp4"

    possible_paths = [
        Path(os.getcwd()) / "uploads" / filename,
        Path(os.getcwd()) / "public" / "uploads" / filename,
        Path(os.getcwd()) / "temp" / filename,
        Path(tempfile.gettempdir()) / filename,
    ]

    for path in possible_paths:
        if path.exists():
            try:
                path.unlink()
                logging.info("✓ Deleted video file: %s", filename)
                return {"success": True, "message": f"Deleted video file: {filename}"}
            except OSError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Video file deletion failed: {e}",
                )

    raise HTTPException(status_code=404, detail="Video file not found")
