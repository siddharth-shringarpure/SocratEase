"""Miscellaneous API routes for health checks and text-to-speech."""
import logging
from typing import Any

import dlib
from flask import Blueprint, jsonify, request, send_file

from api.core.models import TTSRequest
from api.services.tts_service import TTSError, synthesise

misc_bp = Blueprint("misc", __name__)


@misc_bp.route("/api/test", methods=["GET"])
def test_endpoint() -> tuple[dict[str, Any], int]:
    """Verify core API functionality.

    Returns:
        Tuple of response dict and HTTP status code
    """
    try:
        dlib.get_frontal_face_detector()

        emotion_model_available = False
        try:
            from deepface import DeepFace  # noqa: F401
            emotion_model_available = True
        except Exception as e:
            logging.warning("Emotion detection unavailable: %s", e)

        return jsonify({
            "status": "ok",
            "message": "Backend API is running",
            "details": {
                "face_detection": True,
                "emotion_model": emotion_model_available,
                "version": "1.0.0",
            },
        }), 200

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Health check failed: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "details": {
                "face_detection": False,
                "emotion_model": False,
                "version": "1.0.0",
            },
        }), 500


@misc_bp.route("/api/tts-core", methods=["POST", "OPTIONS"])
def tts_core_endpoint() -> Any:
    """Generate speech audio from text.

    Accepts JSON with:
        text: Text to convert (required)
        voice: Voice ID (optional)
        speed: Speech speed 0.7--2.0 (optional)
        category: Practice category forwarded in response header (optional)

    Returns:
        WAV audio file or error JSON
    """
    if request.method == "OPTIONS":
        response = misc_bp.make_default_options_response()
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        })
        return response

    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400

        req = TTSRequest.from_dict(data)

        logging.info("TTS request: %d chars", len(req.text))
        buffer = synthesise(req.text, req.voice, req.speed)

        response = send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="tts_speech.wav",
        )
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, X-Practice-Category"
            ),
        })
        if req.category:
            response.headers["X-Practice-Category"] = req.category

        return response

    except TTSError as e:
        logging.error("TTS error: %s", e)
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, e.status_code

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("TTS endpoint error: %s", e, exc_info=True)
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500
