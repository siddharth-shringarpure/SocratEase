"""
Routes for transcription-related endpoints.

This module provides Flask routes for handling audio transcription requests,
including file upload, processing, analysis, and cleanup functionality.
"""
from flask import Blueprint, request, jsonify
import os
import datetime
import logging
import tempfile
from werkzeug.utils import secure_filename
from api.services.transcription import transcribe_audio
from api.services.text_analysis import (
    calculate_ttr,
    logical_flow,
    analyse_filler_words
)
from api.utils.file_utils import allowed_audio_file

# Create blueprint and logger
transcription_bp = Blueprint('transcription', __name__)
logger = logging.getLogger(__name__)


@transcription_bp.route("/api/speech2text", methods=['POST'])
def transcribe_request() -> tuple[dict, int]:
    """
    Handle audio transcription requests and perform analysis.

    Accepts either a JSON request with an audio filename or a direct file upload.
    Transcribes the audio and performs analysis on the text.

    Returns:
        tuple: (JSON response dict, HTTP status code)
    """
    temp_path: str | None = None

    try:
        logger.info(f"{datetime.datetime.now()}: === Starting transcription request ===")

        # Handle JSON request with filename
        if request.is_json:
            data = request.get_json()
            logger.info(f"Received JSON request: {data}")

            if not data or 'audioFilename' not in data:
                return {
                    "success": False,
                    "error": "Missing audioFilename in request"
                }, 400

            # Process filename and find audio file
            filename = data['audioFilename']
            base_filename = (
                filename if filename.endswith('.wav')
                else f"{filename}.wav"
            )

            # Check primary and fallback paths
            file_paths = [
                os.path.join(os.getcwd(), 'uploads', base_filename),
                os.path.join(os.getcwd(), 'public', 'uploads', base_filename)
            ]

            for path in file_paths:
                if os.path.exists(path):
                    temp_path = path
                    break

            if not temp_path:
                return {
                    "success": False,
                    "error": "Audio file not found"
                }, 404

        # Handle direct file upload
        else:
            if 'file' not in request.files:
                return {
                    "success": False,
                    "error": "No file uploaded"
                }, 400

            file = request.files['file']
            if not file.filename:
                return {
                    "success": False,
                    "error": "Empty filename"
                }, 400

            logger.info(f"{datetime.datetime.now()}: Received file: {file.filename}")
            logger.info(f"{datetime.datetime.now()}: File content type: {file.content_type}")
            
            if not allowed_audio_file(file.filename):
                return {
                    "success": False,
                    "error": "Invalid file type"
                }, 400

            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)

        logger.info(f"{datetime.datetime.now()}: Processing file at: {temp_path}")

        # Transcribe the audio
        try:
            transcription = transcribe_audio(temp_path)
            if transcription:
                # Analyse the transcription
                analysis = analyse_filler_words(transcription)
                
                # Add TTR analysis
                analysis["ttr_analysis"] = calculate_ttr(transcription)
                
                # Attempt to calculate logical flow score
                try:
                    flow_result = logical_flow(transcription)
                    if flow_result > 0:
                        analysis["logical_flow"]["score"] = flow_result
                except Exception as flow_error:
                    logger.warning(f"Using fallback logical flow score: {flow_error}")
                
                # After successful transcription, clean up the original audio file if it's in uploads directory
                # Only for files that were not directly uploaded in this request
                if temp_path and '/uploads/' in temp_path and not '/temp/' in temp_path and os.path.exists(temp_path):
                    try:
                        # Get the original audio filename to log what's being deleted
                        audio_filename = os.path.basename(temp_path)
                        logger.info(f"Cleaning up audio file after successful transcription: {audio_filename}")
                        
                        # Delete the file
                        os.remove(temp_path)
                        logger.info(f"Successfully deleted audio file: {audio_filename}")
                    except Exception as cleanup_error:
                        logger.error(f"Error cleaning up audio file: {str(cleanup_error)}")
                        # Continue even if cleanup fails - the transcription was successful
                
                return jsonify({
                    "success": True,
                    "text": transcription,
                    "analysis": analysis,
                    "cleanup_success": not os.path.exists(temp_path) if temp_path else False
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to transcribe audio"
                }), 500
            
        except Exception as e:
            logger.error(f"{datetime.datetime.now()}: Error processing audio: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Error processing audio: {str(e)}"
            }), 500

    finally:
        # Clean up temp upload file
        if temp_path and '/temp/' in temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Temp file cleanup failed: {e}")


@transcription_bp.route("/api/cleanup-audio", methods=['POST'])
def cleanup_audio_file() -> tuple[dict, int]:
    """
    Clean up audio files from the server.

    Accepts a JSON request with audioFilename and removes the file
    from possible storage locations.

    Returns:
        tuple: (JSON response dict, HTTP status code)
    """
    try:
        if not request.is_json:
            return {
                "success": False,
                "error": "Request must be JSON"
            }, 400

        data = request.get_json()
        logger.info(f"Received cleanup request: {data}")
        
        if not data or 'audioFilename' not in data:
            return {
                "success": False,
                "error": "Missing audioFilename"
            }, 400

        filename = data['audioFilename']
        if not filename.endswith('.wav'):
            filename = f"{filename}.wav"

        # Check all possible file locations
        possible_paths = [
            os.path.join(os.getcwd(), 'uploads', filename),
            os.path.join(os.getcwd(), 'public', 'uploads', filename),
            os.path.join(os.getcwd(), 'temp', filename),
            os.path.join(tempfile.gettempdir(), filename)
        ]
        
        logger.info(f"Checking for audio file {filename} in the following locations: {possible_paths}")
        
        deleted = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"Deleting audio file: {path}")
                    os.remove(path)
                    deleted = True
                    logger.info(f"Successfully deleted audio file: {filename}")
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

        return {
            "success": False,
            "error": "Audio file not found"
        }, 404

    except Exception as e:
        return {
            "success": False,
            "error": "Error cleaning up audio file: " + str(e)
        }, 500


@transcription_bp.route("/api/cleanup-video", methods=['POST'])
def cleanup_video_file() -> tuple[dict, int]:
    """
    Clean up video files from the server.

    Accepts a JSON request with videoFilename and removes the file
    from possible storage locations.

    Returns:
        tuple: (JSON response dict, HTTP status code)
    """
    try:
        if not request.is_json:
            return {
                "success": False,
                "error": "Request must be JSON"
            }, 400

        data = request.get_json()
        if not data or 'videoFilename' not in data:
            return {
                "success": False,
                "error": "Missing videoFilename"
            }, 400

        filename = data['videoFilename']
        if not filename.endswith('.mp4'):
            filename = f"{filename}.mp4"

        # Check all possible file locations
        possible_paths = [
            os.path.join(os.getcwd(), 'uploads', filename),
            os.path.join(os.getcwd(), 'public', 'uploads', filename),
            os.path.join(os.getcwd(), 'temp', filename),
            os.path.join(tempfile.gettempdir(), filename)
        ]

        deleted = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logger.info(f"Deleting video file: {path}")
                    os.remove(path)
                    deleted = True
                    logger.info(f"Successfully deleted video file: {filename}")
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

        return {
            "success": False,
            "error": "File not found"
        }, 404

    except Exception as e:
        return {
            "success": False,
            "error": "Error cleaning up video file: " + str(e)
        }, 500