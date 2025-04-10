"""
Miscellaneous API routes for handling test endpoints and text-to-speech functionality.

This module provides Flask routes for health checks and text-to-speech generation
using the Neuphonic API. It includes error handling and logging.
"""
from flask import Blueprint, jsonify, request, send_file
import os
import traceback
import dlib
import io
import json
import logging
from typing import Tuple, Optional, Dict, Any, Union, BinaryIO

# Set up logging
logger = logging.getLogger(__name__)

# Create blueprint for misc routes
misc_bp = Blueprint('misc', __name__)

# Try importing pyneuphonic with error handling
try:
    from pyneuphonic import Neuphonic, save_audio
    NEUPHONIC_AVAILABLE = True
    logger.info("Successfully imported pyneuphonic")
except Exception as e:
    NEUPHONIC_AVAILABLE = False
    logger.error(f"Failed to import pyneuphonic: {str(e)}")
    traceback.print_exc()


@misc_bp.route("/api/test", methods=['GET'])
def test_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Health check endpoint that verifies core API functionality.

    Returns:
        Tuple containing response dict and HTTP status code
    """
    try:
        # Test core face detection functionality
        detector = dlib.get_frontal_face_detector()

        # Check if emotion detection is available
        emotion_model_available = False
        try:
            import fer
            emotion_model_available = True
            logger.info("Emotion detection available (FER successfully imported)")
        except Exception as e:
            logger.warning(f"FER import failed, emotion detection unavailable: {str(e)}")

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
        logger.error(f"Health check failed: {str(e)}")
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
    voice_id: Optional[str] = None,
    speed: float = 1.0
) -> Tuple[Optional[BinaryIO], Optional[Dict[str, Any]], int]:
    """
    Generate speech using the Neuphonic API.

    Args:
        text: Text to convert to speech
        voice_id: Optional voice ID to use
        speed: Speech speed multiplier (0.5-3.0)

    Returns:
        Tuple containing:
        - Audio buffer (if successful) or None
        - Error dict (if failed) or None  
        - HTTP status code
    """
    try:
        logger.info(f"Starting TTS generation, text length: {len(text)}")

        # Check if TTS service is available
        if not NEUPHONIC_AVAILABLE:
            logger.error("pyneuphonic not available")
            return None, {"error": "TTS service unavailable"}, 500

        # Validate input text
        if not text or not text.strip():
            return None, {"error": "Text cannot be empty"}, 400

        # Warn about long text that may hit API limits
        if len(text) > 5000:
            logger.warning(f"Long text detected ({len(text)} chars)")

        # Use default voice if none provided
        voice_id = voice_id or "f8698a9e-947a-43cd-a897-57edd4070a78"

        # Validate and clamp speed parameter
        try:
            speed = float(speed)
            if not 0.5 <= speed <= 3.0:
                logger.warning("Speed outside valid range, using default")
                speed = 1.0
        except (ValueError, TypeError):
            logger.warning("Invalid speed value, using default")
            speed = 1.0

        # Check for API key in environment
        api_key = os.environ.get('NEUPHONIC_API_KEY')
        if not api_key:
            logger.error("Missing Neuphonic API key")
            return None, {"error": "TTS API key not configured"}, 500

        # Initialise Neuphonic client
        try:
            client = Neuphonic(api_key=api_key)
            logger.info("Neuphonic client ready")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialise Neuphonic client: {error_msg}")
            
            if 'invalid' in error_msg.lower() and 'api key' in error_msg.lower():
                return None, {
                    "error": "Invalid or expired API key",
                    "detailed_error": error_msg,
                    "error_data": {"reason": "api_key_invalid"}
                }, 401
            
            return None, {"error": error_msg}, 500

        # Generate speech
        try:
            # Set up SSE client
            logger.info("Setting up Neuphonic SSE client")
            sse = client.tts.SSEClient()
            logger.info("SSE client ready")
            sse.speed = speed
            sse.voice = voice_id    
            logger.info(f"Sending TTS request for {len(text)} characters")
            # Send TTS request
            response = sse.send(text)
            logger.info("TTS request completed")
            
            # Save audio to buffer
            logger.info("Saving audio to buffer")
            temp_buffer = io.BytesIO()
            save_audio(response, temp_buffer)
            temp_buffer.seek(0)

            # Verify audio was generated
            if not temp_buffer.getbuffer().nbytes:
                logger.error("Generated empty audio")
                return None, {"error": "Empty audio generated"}, 500

            return temp_buffer, None, 200

        except Exception as e:
            error_msg = str(e)
            logger.error(f"TTS generation failed: {error_msg}")

            # Determine error type and status code
            if any(x in error_msg.lower() for x in ['quota', 'limit', 'credit']):
                error_data = {'reason': 'quota_exceeded'}
                status_code = 403
            elif any(x in error_msg.lower() for x in ['api key', 'apikey', 'auth']):
                error_data = {'reason': 'api_key_invalid'}
                status_code = 401
            else:
                error_data = {'reason': 'server_error'}
                status_code = 500

            return None, {
                "error": error_msg,
                "detailed_error": error_msg,
                "error_data": error_data
            }, status_code

    except Exception as e:
        logger.error(f"TTS processing error: {str(e)}")
        return None, {"error": str(e)}, 500


@misc_bp.route("/api/tts-core", methods=['POST', 'OPTIONS'])
def tts_core_endpoint() -> Union[Dict[str, Any], Any]:
    """
    Central endpoint for text-to-speech generation.

    Accepts:
        JSON with:
        - text: Text to convert (required)
        - voice_id: Voice to use (optional)
        - speed: Speech speed (optional)
        - category: Practice category (optional)

    Returns:
        Audio file or error JSON with appropriate status code
    """
    if request.method == 'OPTIONS':
        response = misc_bp.make_default_options_response()
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type, Accept',
            'Access-Control-Allow-Methods': 'POST, OPTIONS'
        })
        return response

    try:
        # Validate request data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing required text parameter"}), 400

        # Extract parameters
        text = data.get('text', '').strip()
        voice_id = data.get('voice')
        speed = data.get('speed', 1.0)
        category = data.get('category')

        logger.info(f"Processing TTS request: {len(text)} chars")

        # Generate speech
        audio_buffer, error_dict, status_code = generate_tts_neuphonic(
            text, voice_id, speed
        )

        if error_dict:
            logger.error(f"TTS generation failed: {error_dict}")
            response = jsonify(error_dict)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response, status_code

        # Prepare successful response
        response = send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='tts_speech.wav'
        )

        # Set CORS and custom headers
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type, Accept',
            'Access-Control-Expose-Headers': 'Content-Type, Content-Disposition, X-Practice-Category'
        })

        if category:
            response.headers['X-Practice-Category'] = category

        return response

    except Exception as e:
        logger.error(f"TTS endpoint error: {str(e)}")
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500