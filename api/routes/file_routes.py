"""
Routes for serving files and handling media uploads.

This module provides Flask routes for serving files from local or remote storage,
handling media uploads (particularly video and audio files), and managing temporary
file cleanup. It includes CORS handling and automatic audio extraction from videos.
"""
import atexit
import datetime
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import traceback
from io import BytesIO

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Blueprint, Response, jsonify, request, send_file

from api.storage import STORAGE_TYPE, UPLOADS_DIR, storage

file_bp = Blueprint("file", __name__)
logger = logging.getLogger(__name__)


@file_bp.route("/uploads/<path:filename>", methods=["OPTIONS"])
def handle_options(filename: str) -> Response:
    """Handle OPTIONS preflight requests for CORS."""
    response = Response()
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    })
    return response


@file_bp.route("/uploads/<path:filename>", methods=["GET"])
def serve_file(filename: str) -> Response:
    """Serve a file from local or remote storage.

    Args:
        filename: Name or path of the file to serve

    Returns:
        Flask Response containing the file or an error message
    """
    logging.info("Attempting to serve file: %s", filename)

    range_header = request.headers.get("Range")

    try:
        possible_paths = [
            os.path.join(UPLOADS_DIR, filename),
            os.path.join(os.getcwd(), "uploads", filename),
            os.path.join(os.getcwd(), "temp", filename),
            os.path.join(tempfile.gettempdir(), filename)
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
                    data,
                    206,
                    mimetype=mime_type,
                    direct_passthrough=True,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(end - start + 1),
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "*"
                    }
                )

            return send_file(local_path, mimetype=mime_type)

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

                return Response(
                    file_bytes,
                    mimetype=mime_type,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "*"
                    }
                )
        else:
            logging.error("StorageManager missing get_file_bytes method")

        logging.error("File not found: %s", filename)
        return "File not found", 404

    except Exception as e:
        logging.error("Error serving file %s: %s", filename, e)
        return f"Error serving file: {e}", 500


@file_bp.route("/api/recordings", methods=["POST"])
def upload_recording() -> Response:
    """Handle recording uploads with automatic audio extraction.

    Expects a video file in the request and extracts audio if present.
    Saves both video and audio files using the storage manager.

    Returns:
        JSON response with file URLs and metadata
    """
    try:
        logging.info(
            "Received request to /api/recordings with method: %s", request.method
        )
        logging.info("Request files: %s", list(request.files.keys()))
        logging.info("Request form data: %s", list(request.form.keys()))

        if "video" not in request.files:
            logging.error("No video file in request")
            return jsonify({"error": "Video file is required"}), 400

        video_file = request.files["video"]
        logging.info(
            "Video info: %s, content type: %s",
            video_file.filename,
            video_file.content_type
        )

        if not video_file.filename:
            return jsonify({"error": "No selected file"}), 400

        try:
            device_id_hash = video_file.filename.split("_")[0]
            if not device_id_hash or len(device_id_hash) < 6:
                return jsonify({"error": "Invalid device ID in filename"}), 400
        except Exception as e:
            logging.error("Error extracting device ID hash: %s", e)
            return jsonify({
                "error": f"Could not extract device ID hash from filename: {video_file.filename}"
            }), 400

        storage.save_file(video_file, video_file.filename, content_type="video/mp4")

        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_video = os.path.join(temp_dir, video_file.filename)

        try:
            video_file.seek(0)
            video_file.save(temp_video)
            if not os.path.exists(temp_video):
                raise IOError(f"Failed to save temp video: {temp_video}")
        except Exception as e:
            logging.error("Error saving to temp: %s", e)
            return jsonify({"error": f"Failed to save temporary video file: {e}"}), 500

        has_audio = False
        audio_filename = None

        probe_result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                temp_video
            ],
            capture_output=True,
            text=True
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
                audio_filename = video_file.filename.replace(".mp4", "_audio.wav")
                temp_audio = os.path.join(temp_dir, audio_filename)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i", temp_video,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ac", "2",
                        "-ar", "44100",
                        "-hide_banner",
                        "-loglevel", "info",
                        temp_audio
                    ],
                    capture_output=True,
                    text=True
                )

                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                    with open(temp_audio, "rb") as audio_file:
                        audio_content = audio_file.read()
                        audio_file_obj = BytesIO(audio_content)
                        storage.save_file(
                            audio_file_obj, audio_filename, content_type="audio/wav"
                        )
                else:
                    has_audio = False
                    audio_filename = None

                for temp_file in [temp_audio, temp_video]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logging.error(
                                "Failed to clean up temp file %s: %s", temp_file, e
                            )

            except Exception as e:
                logging.error("Audio extraction error: %s", e)
                logging.error(traceback.format_exc())
                has_audio = False
                audio_filename = None

        if os.path.exists(temp_video):
            try:
                logging.info("Cleaning up temporary video file: %s", temp_video)
                os.remove(temp_video)
            except Exception as e:
                logging.warning(
                    "Failed to clean up temporary video file: %s", e
                )

        video_url = storage.get_file_url(video_file.filename)
        audio_url = storage.get_file_url(audio_filename) if audio_filename else None

        return jsonify({
            "success": True,
            "filename": video_file.filename,
            "audio_filename": audio_filename,
            "has_audio": has_audio,
            "videoUrl": video_url,
            "audioUrl": audio_url
        })

    except Exception as e:
        logging.error("Recording upload error: %s", e)
        logging.error(traceback.format_exc())
        return jsonify({"error": "Failed to upload recording"}), 500


@file_bp.route("/api/recordings", methods=["GET"])
def get_recording() -> Response:
    """Retrieve a recording file by filename.

    Returns:
        The requested file or an error response
    """
    try:
        filename = request.args.get("filename")
        if not filename:
            return jsonify({"error": "Filename required"}), 400

        device_id_hash = filename.split("_")[0]
        if not device_id_hash or len(device_id_hash) < 6:
            return jsonify({"error": "Invalid device ID"}), 400

        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        if filename.endswith(".mp4"):
            content_type = "video/mp4"
        elif filename.endswith(".wav"):
            content_type = "audio/wav"
        else:
            content_type = "application/octet-stream"

        return send_file(
            file_path,
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )

    except Exception as e:
        logging.error("Recording retrieval error: %s", e)
        logging.error(traceback.format_exc())
        return jsonify({"error": "Failed to retrieve recording"}), 500


@file_bp.route("/api/recordings", methods=["OPTIONS"])
def handle_recordings_options() -> Response:
    """Handle CORS preflight for the /api/recordings endpoint."""
    response = Response()
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true"
    })
    return response


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
            except Exception as e:
                logging.error("Error listing files in %s: %s", temp_dir, e)
                continue

            logging.info(
                "Found %d MP4 and %d WAV files in %s",
                len(mp4_files),
                len(wav_files),
                temp_dir
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
                                age_hours
                            )
                            os.remove(file_path)
                            deleted_count += 1
                            overall_deleted_count += 1
                        except Exception as e:
                            logging.error("Error deleting file %s: %s", file_path, e)
                except Exception as e:
                    logging.error("Error processing file %s: %s", file_path, e)

            logging.info(
                "Cleanup complete for %s. Deleted %d old temporary files.",
                temp_dir,
                deleted_count
            )
        except Exception as e:
            logging.error("Error cleaning up directory %s: %s", temp_dir, e)

    logging.info("Overall cleanup complete. Total deleted: %d files.", overall_deleted_count)


def cleanup_on_startup() -> None:
    """Delete all video and audio files from temp and upload directories on startup."""
    overall_deleted_count = 0

    directories_to_clean = [
        tempfile.gettempdir(),
        os.path.join(os.getcwd(), "temp"),
        os.path.join(os.getcwd(), "uploads"),
        os.path.join(os.getcwd(), "public", "uploads")
    ]

    for directory in directories_to_clean:
        try:
            if not os.path.exists(directory):
                logging.info(
                    "Directory does not exist, skipping cleanup: %s", directory
                )
                continue

            logging.info(
                "Cleaning up all media files on startup in: %s", directory
            )

            media_files = []

            try:
                for filename in os.listdir(directory):
                    if filename.endswith((".mp4", ".wav", "_audio.wav")):
                        file_path = os.path.join(directory, filename)
                        if os.path.isfile(file_path):
                            media_files.append(file_path)
            except Exception as e:
                logging.error("Error listing files in %s: %s", directory, e)
                continue

            logging.info(
                "Found %d media files in %s", len(media_files), directory
            )

            deleted_count = 0
            for file_path in media_files:
                try:
                    logging.info("Deleting media file on startup: %s", file_path)
                    os.remove(file_path)
                    deleted_count += 1
                    overall_deleted_count += 1
                except Exception as e:
                    logging.error("Error deleting file %s: %s", file_path, e)

            logging.info(
                "Startup cleanup complete for %s. Deleted %d media files.",
                directory,
                deleted_count
            )
        except Exception as e:
            logging.error("Error cleaning up directory %s: %s", directory, e)

    logging.info(
        "Startup cleanup complete. Total deleted: %d files.", overall_deleted_count
    )


logging.info("Running startup cleanup...")
cleanup_on_startup()

scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_temp_files, "interval", hours=1)
scheduler.start()

atexit.register(lambda: scheduler.shutdown(wait=False))
