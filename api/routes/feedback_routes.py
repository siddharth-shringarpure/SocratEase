"""
Routes for audio feedback generation.

This module provides endpoints for generating audio feedback using text-to-speech
and speech analysis services. It handles audio file processing, transcription,
and feedback generation.
"""
import datetime
import io
import json
import logging
import os
import subprocess
import traceback
import uuid

from flask import Blueprint, current_app, jsonify, request, send_file

from api.feedback_templates import (
    CATEGORY_FEEDBACK_TEMPLATES,
    DEFAULT_FEEDBACK_TEXT,
    generate_feedback_text,
)
from api.services.text_analysis import analyse_filler_words
from api.services.transcription import transcribe_audio

feedback_bp = Blueprint("feedback", __name__)
logger = logging.getLogger(__name__)

# Import after blueprint creation to avoid circular imports
from api.routes.misc_routes import generate_tts_neuphonic  # noqa: E402

try:
    from pyneuphonic import Neuphonic, save_audio
    NEUPHONIC_AVAILABLE = True
    logging.info("Successfully imported pyneuphonic")
except Exception as e:
    NEUPHONIC_AVAILABLE = False
    logging.error("Failed to import pyneuphonic: %s", e)
    traceback.print_exc()


@feedback_bp.route("/api/tts-test", methods=["GET"])
def test_tts_api() -> tuple[any, int]:
    """Test the Neuphonic API connection.

    Returns:
        Tuple of JSON response and HTTP status code
    """
    try:
        if not NEUPHONIC_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "Neuphonic library not available",
                "details": "Failed to import pyneuphonic library"
            }), 500

        api_key = os.environ.get("NEUPHONIC_API_KEY")
        if not api_key:
            return jsonify({
                "status": "error",
                "message": "API key not configured",
                "details": "NEUPHONIC_API_KEY environment variable not set"
            }), 500

        if len(api_key) < 100:
            return jsonify({
                "status": "warning",
                "message": "API key appears invalid",
                "details": f"API key length seems short ({len(api_key)} chars)"
            }), 200

        try:
            client = Neuphonic(api_key=api_key)

            try:
                logging.info("Testing API by listing voices")
                voices_response = client.voices.list()

                response_info = {}
                if hasattr(voices_response, "status_code"):
                    response_info["status_code"] = voices_response.status_code

                voice_count = 0
                if (
                    hasattr(voices_response, "data")
                    and isinstance(voices_response.data, list)
                ):
                    voice_count = len(voices_response.data)
                    response_info["voice_count"] = voice_count
                    logging.info("Found %d voices in API response", voice_count)
                else:
                    logging.info(
                        "Voice list returned, but couldn't determine count: %s",
                        type(voices_response)
                    )
                    response_info["response_type"] = str(type(voices_response))

                logging.info("Testing TTS with short message")
                sse = client.tts.SSEClient()
                test_text = "API key verification test."
                tts_response = sse.send("a")

                try:
                    temp_buffer = io.BytesIO()
                    save_audio(tts_response, temp_buffer)
                    audio_size = temp_buffer.getbuffer().nbytes
                    audio_saved = audio_size > 0
                    response_info["audio_size"] = audio_size
                    response_info["audio_saved"] = audio_saved
                    logging.info(
                        "Successfully saved test audio (%d bytes)", audio_size
                    )
                except Exception as save_error:
                    response_info["audio_error"] = str(save_error)
                    logging.warning("Failed to save test audio: %s", save_error)

                return jsonify({
                    "status": "success",
                    "message": "Neuphonic API key is valid and working",
                    "details": {
                        "key_length": len(api_key),
                        "key_prefix": (
                            api_key[:4] + "..." if len(api_key) > 8 else None
                        ),
                        "voice_count": voice_count,
                        "test_text": test_text,
                        "response_info": response_info
                    }
                }), 200

            except Exception as operation_error:
                error_message = str(operation_error)

                if any(x in error_message.lower() for x in ["quota", "limit"]):
                    return jsonify({
                        "status": "error",
                        "message": "API key is valid but quota has been exceeded",
                        "details": error_message
                    }), 403
                elif any(
                    x in error_message.lower()
                    for x in ["unauthorized", "authentication", "401"]
                ):
                    return jsonify({
                        "status": "error",
                        "message": "API key is invalid or expired",
                        "details": error_message
                    }), 401
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Failed to perform TTS operation",
                        "details": error_message
                    }), 500

        except Exception as client_error:
            error_message = str(client_error)
            if "invalid" in error_message.lower() and "api key" in error_message.lower():
                return jsonify({
                    "status": "error",
                    "message": "API key is invalid",
                    "details": error_message
                }), 401
            else:
                return jsonify({
                    "status": "error",
                    "message": "Failed to initialise Neuphonic client",
                    "details": error_message
                }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Unexpected error during TTS test",
            "details": str(e)
        }), 500


@feedback_bp.route("/api/tts", methods=["POST", "OPTIONS"])
def generate_tts() -> any:
    """Generate text-to-speech audio using the core TTS function.

    Returns:
        Audio file response or error JSON
    """
    if request.method == "OPTIONS":
        response = current_app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    try:
        logging.info("Received TTS request, passing to core TTS function")

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing required parameter: text"}), 400

        text = data.get("text", "").strip()
        voice_id = data.get("voice")
        speed = data.get("speed", 1.0)

        audio_buffer, error_dict, status_code = generate_tts_neuphonic(
            text, voice_id, speed
        )

        if error_dict:
            error_response = jsonify(error_dict)
            error_response.headers["Access-Control-Allow-Origin"] = "*"
            return error_response, status_code

        response = send_file(
            audio_buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="tts_speech.wav"
        )

        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Expose-Headers"] = (
            "Content-Type, Content-Disposition"
        )

        logging.info("Successfully generated and returned TTS audio")
        return response

    except Exception as e:
        error_message = str(e)
        logging.error("TTS endpoint error: %s", error_message)
        traceback.print_exc()
        error_response = jsonify({"error": error_message})
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response, 500


@feedback_bp.route("/api/audio-feedback", methods=["POST", "OPTIONS"])
def generate_audio_feedback() -> any:
    """Generate audio feedback using speech analysis and TTS.

    Returns:
        Audio feedback file response or error JSON
    """
    if request.method == "OPTIONS":
        response = current_app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    try:
        logging.info("Starting audio feedback generation")

        default_text = DEFAULT_FEEDBACK_TEXT
        speech_speed = 1.0

        practice_category = None
        try:
            if request.form and "category" in request.form:
                practice_category = request.form.get("category")
            elif request.args and "category" in request.args:
                practice_category = request.args.get("category")
            logging.info("Practice category: %s", practice_category)
        except Exception as e:
            logging.warning("Error parsing category: %s", e)

        analysis = None
        text = ""

        try:
            if "file" in request.files and request.files["file"].filename:
                temp_dir = os.path.join(os.getcwd(), "temp")
                os.makedirs(temp_dir, exist_ok=True)

                unique_id = (
                    datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    + "_"
                    + str(uuid.uuid4())[:8]
                )
                temp_input = os.path.join(temp_dir, f"input_{unique_id}.wav")
                temp_audio = os.path.join(temp_dir, f"audio_{unique_id}.mp3")

                file = request.files["file"]
                logging.info(
                    "Received file: %s, mimetype: %s", file.filename, file.content_type
                )

                file.save(temp_input)
                input_size = os.path.getsize(temp_input)
                logging.info(
                    "Saved input file to %s (size: %d bytes)", temp_input, input_size
                )

                if input_size > 0:
                    logging.info("Converting audio for transcription")
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_input,
                        "-vn",
                        "-acodec", "libmp3lame",
                        "-ar", "16000",
                        "-ac", "1",
                        "-b:a", "192k",
                        "-af", "highpass=f=50,lowpass=f=15000,volume=2,afftdn=nf=-20",
                        temp_audio
                    ]

                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)

                        if (
                            os.path.exists(temp_audio)
                            and os.path.getsize(temp_audio) > 0
                        ):
                            logging.info("Transcribing audio")
                            text = transcribe_audio(temp_audio)

                            if text:
                                logging.info(
                                    "Analysing transcribed text (%d chars)", len(text)
                                )
                                analysis = analyse_filler_words(text)

                    except Exception as conversion_error:
                        logging.error(
                            "Audio processing error: %s", conversion_error
                        )

                    finally:
                        for temp_file in [temp_input, temp_audio]:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception as e:
                                logging.error(
                                    "Failed to clean up %s: %s", temp_file, e
                                )

        except Exception as file_error:
            logging.error("File processing error: %s", file_error)

        feedback_text = generate_feedback_text(
            analysis=analysis,
            practice_category=practice_category,
            default_text=default_text
        )

        logging.info(
            "Generating TTS for feedback (%d chars)", len(feedback_text)
        )

        audio_buffer, error_dict, status_code = generate_tts_neuphonic(
            text=feedback_text,
            speed=speech_speed
        )

        if error_dict:
            error_response = jsonify(error_dict)
            error_response.headers["Access-Control-Allow-Origin"] = "*"
            return error_response, status_code

        logging.info("Creating send_file response with analysis data in headers")
        response = send_file(
            audio_buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="feedback_speech.wav"
        )

        metrics_json = json.dumps(analysis) if analysis else "{}"

        if analysis and feedback_text:
            analysis_dict = analysis if isinstance(analysis, dict) else {}
            analysis_dict["feedback_text"] = feedback_text
            metrics_json = json.dumps(analysis_dict)

        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, "
                "X-Speech-Metrics, X-Practice-Category"
            ),
            "X-Speech-Metrics": metrics_json,
            "X-Practice-Category": practice_category or "unknown"
        })

        logging.info("Successfully generated and returned audio feedback")
        return response

    except Exception as e:
        error_message = str(e)
        logging.error("Feedback generation error: %s", error_message)
        traceback.print_exc()
        error_response = jsonify({"error": error_message})
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response, 500


@feedback_bp.route("/api/generate-feedback-text", methods=["POST", "OPTIONS"])
def generate_feedback_text_route() -> any:
    """Generate feedback text without TTS for frontend caching.

    Returns:
        JSON response containing the feedback text
    """
    if request.method == "OPTIONS":
        response = current_app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    try:
        logging.info("Starting feedback text generation")

        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        analysis = data.get("analysis")
        category = data.get("category")

        logging.info("Generating feedback for category: %s", category)

        feedback_text = generate_feedback_text(
            analysis=analysis,
            practice_category=category,
            default_text=DEFAULT_FEEDBACK_TEXT
        )

        response = jsonify({
            "feedback_text": feedback_text,
            "category": category
        })

        response.headers["Access-Control-Allow-Origin"] = "*"
        logging.info("Generated feedback text (%d chars)", len(feedback_text))
        return response

    except Exception as e:
        error_message = str(e)
        logging.error("Feedback text generation error: %s", error_message)
        traceback.print_exc()
        error_response = jsonify({"error": error_message})
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        return error_response, 500
