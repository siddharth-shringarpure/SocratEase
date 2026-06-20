"""
Routes for transcription-related endpoints.

This module provides Flask routes for handling audio transcription requests,
including file upload, processing, analysis, and cleanup functionality.
"""
import datetime
import logging
import os
import tempfile

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from api.services.text_analysis import analyse_filler_words, calculate_ttr, logical_flow
from api.services.transcription import transcribe_audio
from api.utils.file_utils import allowed_audio_file

transcription_bp = Blueprint("transcription", __name__)
logger = logging.getLogger(__name__)


@transcription_bp.route("/api/speech2text", methods=["POST"])
def transcribe_request() -> tuple[dict, int]:
    """Handle audio transcription requests and perform analysis.

    Accepts either a JSON request with an audio filename or a direct file upload.
    Transcribes the audio and performs analysis on the text.

    Returns:
        Tuple of JSON response dict and HTTP status code
    """
    temp_path: str | None = None

    try:
        logging.info(
            "%s: === Starting transcription request ===",
            datetime.datetime.now()
        )

        if request.is_json:
            data = request.get_json()
            logging.info("Received JSON request: %s", data)

            if not data or "audioFilename" not in data:
                return {
                    "success": False,
                    "error": "Missing audioFilename in request"
                }, 400

            filename = data["audioFilename"]
            base_filename = (
                filename if filename.endswith(".wav")
                else f"{filename}.wav"
            )

            file_paths = [
                os.path.join(os.getcwd(), "uploads", base_filename),
                os.path.join(os.getcwd(), "public", "uploads", base_filename)
            ]

            for path in file_paths:
                if os.path.exists(path):
                    temp_path = path
                    break

            if not temp_path:
                return {"success": False, "error": "Audio file not found"}, 404

        else:
            if "file" not in request.files:
                return {"success": False, "error": "No file uploaded"}, 400

            file = request.files["file"]
            if not file.filename:
                return {"success": False, "error": "Empty filename"}, 400

            logging.info(
                "%s: Received file: %s", datetime.datetime.now(), file.filename
            )
            logging.info(
                "%s: File content type: %s",
                datetime.datetime.now(),
                file.content_type
            )

            if not allowed_audio_file(file.filename):
                return {"success": False, "error": "Invalid file type"}, 400

            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)

        logging.info(
            "%s: Processing file at: %s", datetime.datetime.now(), temp_path
        )

        try:
            transcription = transcribe_audio(temp_path)
            if transcription:
                analysis = analyse_filler_words(transcription)
                analysis["ttr_analysis"] = calculate_ttr(transcription)

                try:
                    flow_result = logical_flow(transcription)
                    if flow_result > 0:
                        analysis["logical_flow"]["score"] = flow_result
                except Exception as flow_error:
                    logging.warning(
                        "Using fallback logical flow score: %s", flow_error
                    )

                # Clean up the source audio once transcription succeeds,
                # but only for files that were pre-uploaded rather than streamed
                if (
                    temp_path
                    and "/uploads/" in temp_path
                    and "/temp/" not in temp_path
                    and os.path.exists(temp_path)
                ):
                    try:
                        audio_filename = os.path.basename(temp_path)
                        logging.info(
                            "Cleaning up audio file after successful transcription: %s",
                            audio_filename
                        )
                        os.remove(temp_path)
                        logging.info(
                            "Successfully deleted audio file: %s", audio_filename
                        )
                    except Exception as cleanup_error:
                        logging.error(
                            "Error cleaning up audio file: %s", cleanup_error
                        )

                return jsonify({
                    "success": True,
                    "text": transcription,
                    "analysis": analysis,
                    "cleanup_success": (
                        not os.path.exists(temp_path) if temp_path else False
                    )
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to transcribe audio"
                }), 500

        except Exception as e:
            logging.error(
                "%s: Error processing audio: %s", datetime.datetime.now(), e
            )
            return jsonify({
                "success": False,
                "error": f"Error processing audio: {e}"
            }), 500

    finally:
        if temp_path and "/temp/" in temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.error("Temp file cleanup failed: %s", e)


@transcription_bp.route("/api/cleanup-audio", methods=["POST"])
def cleanup_audio_file() -> tuple[dict, int]:
    """Clean up audio files from the server.

    Accepts a JSON request with audioFilename and removes the file
    from possible storage locations.

    Returns:
        Tuple of JSON response dict and HTTP status code
    """
    try:
        if not request.is_json:
            return {"success": False, "error": "Request must be JSON"}, 400

        data = request.get_json()
        logging.info("Received cleanup request: %s", data)

        if not data or "audioFilename" not in data:
            return {"success": False, "error": "Missing audioFilename"}, 400

        filename = data["audioFilename"]
        if not filename.endswith(".wav"):
            filename = f"{filename}.wav"

        possible_paths = [
            os.path.join(os.getcwd(), "uploads", filename),
            os.path.join(os.getcwd(), "public", "uploads", filename),
            os.path.join(os.getcwd(), "temp", filename),
            os.path.join(tempfile.gettempdir(), filename)
        ]

        logging.info(
            "Checking for audio file %s in locations: %s",
            filename,
            possible_paths
        )

        deleted = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logging.info("Deleting audio file: %s", path)
                    os.remove(path)
                    deleted = True
                    logging.info("Successfully deleted audio file: %s", filename)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Audio file deletion failed: {e}"
                    }, 500

        if deleted:
            return {
                "success": True,
                "message": f"Successfully deleted audio file: {filename}"
            }, 200

        return {"success": False, "error": "Audio file not found"}, 404

    except Exception as e:
        return {
            "success": False,
            "error": "Error cleaning up audio file: " + str(e)
        }, 500


@transcription_bp.route("/api/cleanup-video", methods=["POST"])
def cleanup_video_file() -> tuple[dict, int]:
    """Clean up video files from the server.

    Accepts a JSON request with videoFilename and removes the file
    from possible storage locations.

    Returns:
        Tuple of JSON response dict and HTTP status code
    """
    try:
        if not request.is_json:
            return {"success": False, "error": "Request must be JSON"}, 400

        data = request.get_json()
        if not data or "videoFilename" not in data:
            return {"success": False, "error": "Missing videoFilename"}, 400

        filename = data["videoFilename"]
        if not filename.endswith(".mp4"):
            filename = f"{filename}.mp4"

        possible_paths = [
            os.path.join(os.getcwd(), "uploads", filename),
            os.path.join(os.getcwd(), "public", "uploads", filename),
            os.path.join(os.getcwd(), "temp", filename),
            os.path.join(tempfile.gettempdir(), filename)
        ]

        deleted = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logging.info("Deleting video file: %s", path)
                    os.remove(path)
                    deleted = True
                    logging.info("Successfully deleted video file: %s", filename)
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Deletion failed: {e}"
                    }, 500

        if deleted:
            return {
                "success": True,
                "message": f"Successfully deleted video file: {filename}"
            }, 200

        return {"success": False, "error": "File not found"}, 404

    except Exception as e:
        return {
            "success": False,
            "error": "Error cleaning up video file: " + str(e)
        }, 500
