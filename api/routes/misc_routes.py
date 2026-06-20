"""
Miscellaneous API routes for health checks and text-to-speech functionality.

This module provides Flask routes for health checks and text-to-speech generation
using the Neuphonic API. It includes error handling and logging.
"""
import io
import logging
import os
import traceback
from typing import Any, BinaryIO

import dlib
from flask import Blueprint, jsonify, request, send_file

logger = logging.getLogger(__name__)

misc_bp = Blueprint("misc", __name__)

try:
    from pyneuphonic import Neuphonic, save_audio
    NEUPHONIC_AVAILABLE = True
    logging.info("Successfully imported pyneuphonic")
except Exception as e:
    NEUPHONIC_AVAILABLE = False
    logging.error("Failed to import pyneuphonic: %s", e)
    traceback.print_exc()


@misc_bp.route("/api/test", methods=["GET"])
def test_endpoint() -> tuple[dict[str, Any], int]:
    """Verify core API functionality.

    Returns:
        Tuple containing response dict and HTTP status code
    """
    try:
        dlib.get_frontal_face_detector()

        emotion_model_available = False
        try:
            from deepface import DeepFace  # noqa: F401
            emotion_model_available = True
            logging.info("Emotion detection available (DeepFace successfully imported)")
        except Exception as e:
            logging.warning("Emotion detection unavailable: %s", e)

        response = {
            "status": "ok",
            "message": "Backend API is running and core dependencies are available",
            "details": {
                "face_detection": True,
                "emotion_model": emotion_model_available,
                "version": "1.0.0"
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error("Health check failed: %s", e)
        traceback.print_exc()

        response = {
            "status": "error",
            "message": str(e),
            "details": {
                "face_detection": False,
                "emotion_model": False,
                "version": "1.0.0"
            }
        }
        return jsonify(response), 500


def generate_tts_neuphonic(
    text: str,
    voice_id: str | None = None,
    speed: float = 1.0
) -> tuple[BinaryIO | None, dict[str, Any] | None, int]:
    """Generate speech using the Neuphonic API.

    Args:
        text: Text to convert to speech
        voice_id: Optional voice ID to use
        speed: Speech speed multiplier (0.5--3.0)

    Returns:
        Tuple of (audio buffer or None, error dict or None, HTTP status code)
    """
    try:
        logging.info("Starting TTS generation, text length: %d", len(text))

        if not NEUPHONIC_AVAILABLE:
            logging.error("pyneuphonic not available")
            return None, {"error": "TTS service unavailable"}, 500

        if not text or not text.strip():
            return None, {"error": "Text cannot be empty"}, 400

        if len(text) > 5000:
            logging.warning("Long text detected (%d chars)", len(text))

        voice_id = voice_id or "f8698a9e-947a-43cd-a897-57edd4070a78"

        try:
            speed = float(speed)
            if not 0.5 <= speed <= 3.0:
                logging.warning("Speed outside valid range, using default")
                speed = 1.0
        except (ValueError, TypeError):
            logging.warning("Invalid speed value, using default")
            speed = 1.0

        api_key = os.environ.get("NEUPHONIC_API_KEY")
        if not api_key:
            logging.error("Missing Neuphonic API key")
            return None, {"error": "TTS API key not configured"}, 500

        try:
            client = Neuphonic(api_key=api_key)
            logging.info("Neuphonic client ready")
        except Exception as e:
            error_msg = str(e)
            logging.error("Failed to initialise Neuphonic client: %s", error_msg)

            if "invalid" in error_msg.lower() and "api key" in error_msg.lower():
                return None, {
                    "error": "Invalid or expired API key",
                    "detailed_error": error_msg,
                    "error_data": {"reason": "api_key_invalid"}
                }, 401

            return None, {"error": error_msg}, 500

        try:
            logging.info("Setting up Neuphonic SSE client")
            sse = client.tts.SSEClient()
            logging.info("SSE client ready")
            sse.speed = speed
            sse.voice = voice_id
            logging.info("Sending TTS request for %d characters", len(text))
            response = sse.send(text)
            logging.info("TTS request completed")

            logging.info("Saving audio to buffer")
            temp_buffer = io.BytesIO()
            save_audio(response, temp_buffer)
            temp_buffer.seek(0)

            if not temp_buffer.getbuffer().nbytes:
                logging.error("Generated empty audio")
                return None, {"error": "Empty audio generated"}, 500

            return temp_buffer, None, 200

        except Exception as e:
            error_msg = str(e)
            logging.error("TTS generation failed: %s", error_msg)

            if any(x in error_msg.lower() for x in ["quota", "limit", "credit"]):
                error_data = {"reason": "quota_exceeded"}
                status_code = 403
            elif any(x in error_msg.lower() for x in ["api key", "apikey", "auth"]):
                error_data = {"reason": "api_key_invalid"}
                status_code = 401
            else:
                error_data = {"reason": "server_error"}
                status_code = 500

            return None, {
                "error": error_msg,
                "detailed_error": error_msg,
                "error_data": error_data
            }, status_code

    except Exception as e:
        logging.error("TTS processing error: %s", e)
        return None, {"error": str(e)}, 500


@misc_bp.route("/api/tts-core", methods=["POST", "OPTIONS"])
def tts_core_endpoint() -> Any:
    """Serve as the central endpoint for text-to-speech generation.

    Accepts:
        JSON with:
        - text: Text to convert (required)
        - voice_id: Voice to use (optional)
        - speed: Speech speed (optional)
        - category: Practice category (optional)

    Returns:
        Audio file or error JSON with appropriate status code
    """
    if request.method == "OPTIONS":
        response = misc_bp.make_default_options_response()
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Allow-Methods": "POST, OPTIONS"
        })
        return response

    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing required text parameter"}), 400

        text = data.get("text", "").strip()
        voice_id = data.get("voice")
        speed = data.get("speed", 1.0)
        category = data.get("category")

        logging.info("Processing TTS request: %d chars", len(text))

        audio_buffer, error_dict, status_code = generate_tts_neuphonic(
            text, voice_id, speed
        )

        if error_dict:
            logging.error("TTS generation failed: %s", error_dict)
            response = jsonify(error_dict)
            response.headers["Access-Control-Allow-Origin"] = "*"
            return response, status_code

        response = send_file(
            audio_buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="tts_speech.wav"
        )

        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, X-Practice-Category"
            )
        })

        if category:
            response.headers["X-Practice-Category"] = category

        return response

    except Exception as e:
        logging.error("TTS endpoint error: %s", e)
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500
