"""Routes for audio feedback generation."""
import datetime
import json
import logging
import os
import subprocess
import uuid
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request, send_file

from api.core.constants import DEFAULT_FEEDBACK_TEXT
from api.services.feedback_service import generate_feedback_text
from api.services.text_analysis_service import analyse_filler_words
from api.services.transcription_service import transcribe_audio
from api.services.tts_service import TTSError, synthesise

feedback_bp = Blueprint("feedback", __name__)


def _options_response() -> Response:
    response = current_app.make_default_options_response()
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Accept",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
    })
    return response


def _error_response(message: str, status_code: int) -> tuple[Response, int]:
    response = jsonify({"error": message})
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response, status_code


@feedback_bp.route("/api/audio-feedback", methods=["POST", "OPTIONS"])
def generate_audio_feedback() -> Any:
    """Generate spoken feedback from an uploaded audio file.

    Accepts multipart form data with:
        file: Audio file to analyse (optional)
        category: Practice category (optional, form or query param)

    Returns:
        WAV audio response with analysis in X-Speech-Metrics header,
        or error JSON
    """
    if request.method == "OPTIONS":
        return _options_response()

    try:
        practice_category = (
            request.form.get("category")
            or request.args.get("category")
        )

        analysis = None
        speech_speed = 1.0

        if "file" in request.files and request.files["file"].filename:
            analysis = _process_uploaded_audio(request.files["file"])

        feedback_text = generate_feedback_text(
            analysis=analysis,
            practice_category=practice_category,
            default_text=DEFAULT_FEEDBACK_TEXT,
        )

        logging.info("Generating TTS for feedback: %d chars", len(feedback_text))
        buffer = synthesise(feedback_text, speed=speech_speed)

        metrics_dict = analysis if isinstance(analysis, dict) else {}
        if feedback_text:
            metrics_dict["feedback_text"] = feedback_text

        response = send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="feedback_speech.wav",
        )
        response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
            "Access-Control-Expose-Headers": (
                "Content-Type, Content-Disposition, "
                "X-Speech-Metrics, X-Practice-Category"
            ),
            "X-Speech-Metrics": json.dumps(metrics_dict),
            "X-Practice-Category": practice_category or "unknown",
        })
        return response

    except TTSError as e:
        logging.error("TTS error in audio feedback: %s", e)
        return _error_response(str(e), e.status_code)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Feedback generation error: %s", e, exc_info=True)
        return _error_response(str(e), 500)


def _process_uploaded_audio(file: Any) -> dict | None:
    """Transcribe and analyse an uploaded audio file.

    Args:
        file: Werkzeug FileStorage object

    Returns:
        Analysis dict from analyse_filler_words, or None on failure
    """
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    unique_id = (
        datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + "_" + str(uuid.uuid4())[:8]
    )
    temp_input = os.path.join(temp_dir, f"input_{unique_id}.wav")
    temp_audio = os.path.join(temp_dir, f"audio_{unique_id}.mp3")

    try:
        file.save(temp_input)
        if not os.path.getsize(temp_input):
            return None

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_input, "-vn",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
            "-b:a", "192k",
            "-af", "highpass=f=50,lowpass=f=15000,volume=2,afftdn=nf=-20",
            temp_audio,
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=30)

        if not (os.path.exists(temp_audio) and os.path.getsize(temp_audio)):
            return None

        text = transcribe_audio(temp_audio)
        if not text:
            return None

        logging.info("Analysing transcribed text: %d chars", len(text))
        return analyse_filler_words(text)

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Audio processing failures are non-fatal; fallback to default feedback
        logging.error("Audio processing error: %s", e)
        return None

    finally:
        for path in [temp_input, temp_audio]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logging.error("Failed to clean up %s: %s", path, e)


@feedback_bp.route("/api/generate-feedback-text", methods=["POST", "OPTIONS"])
def generate_feedback_text_route() -> Any:
    """Generate feedback text without TTS for frontend caching.

    Returns:
        JSON with feedback_text and category fields
    """
    if request.method == "OPTIONS":
        return _options_response()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        analysis = data.get("analysis")
        category = data.get("category")

        feedback_text = generate_feedback_text(
            analysis=analysis,
            practice_category=category,
            default_text=DEFAULT_FEEDBACK_TEXT,
        )

        logging.info("Generated feedback text: %d chars", len(feedback_text))
        response = jsonify({"feedback_text": feedback_text, "category": category})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Feedback text generation error: %s", e, exc_info=True)
        return _error_response(str(e), 500)
