"""Router for file serving and media upload endpoints."""
import atexit
import datetime
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

# TODO: storage dependency — pending proper storage service extraction
from api.storage import STORAGE_TYPE, UPLOADS_DIR, storage

file_router = APIRouter()


@file_router.api_route("/uploads/{filename:path}", methods=["GET", "HEAD"])
async def serve_file(filename: str, request: Request) -> Response:
    """Serve a file from local or remote storage.

    Args:
        filename: Name or path of the file to serve
        request: Incoming request (used to read Range header)

    Returns:
        File response, partial (206) if Range requested

    Raises:
        HTTPException: If file not found or serving fails
    """
    logging.info("Attempting to serve file: %s", filename)
    range_header = request.headers.get("Range")

    try:
        possible_paths = [
            os.path.join(UPLOADS_DIR, filename),
            os.path.join(os.getcwd(), "uploads", filename),
            os.path.join(os.getcwd(), "temp", filename),
            os.path.join(tempfile.gettempdir(), filename),
        ]

        local_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                local_path = path
                logging.info("Found file locally at: %s", local_path)
                break

        if local_path:
            file_size = os.path.getsize(local_path)
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

            logging.info("Serving local file: %s, mime type: %s", local_path, mime_type)

            if range_header:
                start, end = 0, file_size - 1

                if range_header.startswith("bytes="):
                    ranges = range_header[6:].split("-")
                    if ranges[0]:
                        start = int(ranges[0])
                    if len(ranges) > 1 and ranges[1]:
                        end = min(int(ranges[1]), file_size - 1)

                with open(local_path, "rb") as f:
                    f.seek(start)
                    data = f.read(end - start + 1)

                return Response(
                    content=data,
                    status_code=206,
                    media_type=mime_type,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(end - start + 1),
                    },
                )

            return FileResponse(local_path, media_type=mime_type)

        if STORAGE_TYPE != "local" and hasattr(storage, "get_file_bytes"):
            logging.info("Attempting remote storage retrieval: %s", filename)
            file_bytes = storage.get_file_bytes(filename)

            if file_bytes:
                mime_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )
                logging.info(
                    "Serving remote file: %s, mime type: %s", filename, mime_type
                )
                return Response(content=file_bytes, media_type=mime_type)
        else:
            logging.error("StorageManager missing get_file_bytes method")

        logging.error("File not found: %s", filename)
        raise HTTPException(status_code=404, detail="File not found")

    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error serving file %s: %s", filename, e)
        raise HTTPException(status_code=500, detail=f"Error serving file: {e}")


@file_router.post("/api/recordings")
async def upload_recording(video: UploadFile = File(...)) -> dict:
    """Handle recording uploads with automatic audio extraction.

    Reads the uploaded video, saves to temp, probes for audio, extracts
    audio with ffmpeg if present, and saves both via the storage manager.

    Args:
        video: Uploaded video file

    Returns:
        Dict with file URLs and metadata

    Raises:
        HTTPException: On validation or processing failure
    """
    try:
        logging.info("Received recording upload: %s (%s)", video.filename, video.content_type)

        if not video.filename:
            raise HTTPException(status_code=400, detail="No selected file")

        try:
            device_id_hash = video.filename.split("_")[0]
            if not device_id_hash or len(device_id_hash) < 6:
                raise HTTPException(status_code=400, detail="Invalid device ID in filename")
        except HTTPException:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error extracting device ID hash: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract device ID hash from filename: {video.filename}",
            )

        contents = await video.read()
        storage.save_file(BytesIO(contents), video.filename, content_type="video/mp4")

        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_video = temp_dir / video.filename
        temp_video.write_bytes(contents)

        has_audio = False
        audio_filename = None

        probe_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                str(temp_video),
            ],
            capture_output=True,
            text=True,
        )

        logging.info("FFprobe return code: %d", probe_result.returncode)

        try:
            probe_data = json.loads(probe_result.stdout)
            has_audio = (
                probe_result.returncode == 0
                and probe_data.get("streams")
                and probe_data["streams"][0].get("codec_type") == "audio"
            )
        except json.JSONDecodeError:
            logging.error("Error parsing ffprobe output")
            has_audio = False

        if has_audio:
            try:
                audio_filename = video.filename.replace(".mp4", "_audio.wav")
                temp_audio = temp_dir / audio_filename

                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(temp_video),
                        "-vn", "-acodec", "pcm_s16le",
                        "-ac", "2", "-ar", "44100",
                        "-hide_banner", "-loglevel", "info",
                        str(temp_audio),
                    ],
                    capture_output=True,
                    text=True,
                )

                if temp_audio.exists() and temp_audio.stat().st_size > 0:
                    storage.save_file(
                        BytesIO(temp_audio.read_bytes()),
                        audio_filename,
                        content_type="audio/wav",
                    )
                else:
                    has_audio = False
                    audio_filename = None

                for tmp in [temp_audio, temp_video]:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError as e:
                        logging.error("Failed to clean up temp file %s: %s", tmp, e)

            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Audio extraction error: %s", e, exc_info=True)
                has_audio = False
                audio_filename = None

        try:
            temp_video.unlink(missing_ok=True)
        except OSError as e:
            logging.warning("Failed to clean up temporary video file: %s", e)

        video_url = storage.get_file_url(video.filename)
        audio_url = storage.get_file_url(audio_filename) if audio_filename else None

        return {
            "success": True,
            "filename": video.filename,
            "audio_filename": audio_filename,
            "has_audio": has_audio,
            "videoUrl": video_url,
            "audioUrl": audio_url,
        }

    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Recording upload error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload recording")


@file_router.get("/api/recordings")
async def get_recording(filename: str = Query(...)) -> FileResponse:
    """Retrieve a recording file by filename.

    Args:
        filename: Name of the recording file

    Returns:
        FileResponse for the requested recording

    Raises:
        HTTPException: If file not found or validation fails
    """
    try:
        device_id_hash = filename.split("_")[0]
        if not device_id_hash or len(device_id_hash) < 6:
            raise HTTPException(status_code=400, detail="Invalid device ID")

        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        if filename.endswith(".mp4"):
            content_type = "video/mp4"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        else:
            content_type = "application/octet-stream"

        return FileResponse(file_path, media_type=content_type, filename=filename)

    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Recording retrieval error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve recording")


def cleanup_old_temp_files() -> None:
    """Delete temporary files older than one hour from temp directories."""
    overall_deleted_count = 0
    temp_dirs = [tempfile.gettempdir(), os.path.join(os.getcwd(), "temp")]

    for temp_dir in temp_dirs:
        try:
            if not os.path.exists(temp_dir):
                logging.info("Directory does not exist, skipping cleanup: %s", temp_dir)
                continue

            logging.info("Cleaning up temporary files in: %s", temp_dir)

            mp4_files = []
            wav_files = []

            try:
                for filename in os.listdir(temp_dir):
                    if filename.endswith(".mp4"):
                        mp4_files.append(os.path.join(temp_dir, filename))
                    elif filename.endswith(".wav") or filename.endswith("_audio.wav"):
                        wav_files.append(os.path.join(temp_dir, filename))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Error listing files in %s: %s", temp_dir, e)
                continue

            logging.info(
                "Found %d MP4 and %d WAV files in %s",
                len(mp4_files),
                len(wav_files),
                temp_dir,
            )

            current_time = datetime.datetime.now()
            deleted_count = 0

            for file_path in mp4_files + wav_files:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    file_time = datetime.datetime.fromtimestamp(file_mtime)
                    age_hours = (current_time - file_time).total_seconds() / 3600

                    if age_hours > 1:
                        try:
                            logging.info(
                                "Deleting old temporary file: %s (age: %.2f hours)",
                                file_path,
                                age_hours,
                            )
                            os.remove(file_path)
                            deleted_count += 1
                            overall_deleted_count += 1
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logging.error("Error deleting file %s: %s", file_path, e)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logging.error("Error processing file %s: %s", file_path, e)

            logging.info(
                "Cleanup complete for %s. Deleted %d old temporary files.",
                temp_dir,
                deleted_count,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error cleaning up directory %s: %s", temp_dir, e)

    logging.info("Overall cleanup complete. Total deleted: %d files.", overall_deleted_count)


def cleanup_on_startup() -> None:
    """Delete temp media files older than 1 hour on startup.

    Only cleans temp directories — uploads are not touched so recordings
    survive server restarts.
    """
    temp_dirs = [tempfile.gettempdir(), os.path.join(os.getcwd(), "temp")]
    cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
    deleted = 0

    for temp_dir in temp_dirs:
        if not os.path.exists(temp_dir):
            continue
        try:
            for filename in os.listdir(temp_dir):
                if not filename.endswith((".mp4", ".wav")):
                    continue
                file_path = os.path.join(temp_dir, filename)
                try:
                    if (
                        os.path.isfile(file_path)
                        and datetime.datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ) < cutoff
                    ):
                        os.remove(file_path)
                        deleted += 1
                except OSError as e:
                    logging.error("Error deleting temp file %s: %s", file_path, e)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Error scanning temp dir %s: %s", temp_dir, e)

    logging.info("Startup temp cleanup complete. Deleted %d old files.", deleted)


logging.info("Running startup cleanup...")
cleanup_on_startup()

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_temp_files, "interval", hours=1)
scheduler.start()

atexit.register(lambda: scheduler.shutdown(wait=False))
