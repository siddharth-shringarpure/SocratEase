"""
Routes for serving files and handling media uploads.

This module provides Flask routes for serving files from local or remote storage,
handling media uploads (particularly video and audio files), and managing temporary
file cleanup. It includes CORS handling and automatic audio extraction from videos.
"""

from flask import Blueprint, request, send_file, Response, jsonify
import os
import mimetypes
import logging
import subprocess
import datetime
import tempfile
from api.storage import storage, STORAGE_TYPE, UPLOADS_DIR
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import json
from io import BytesIO
import traceback

# Initialise blueprint and logger
file_bp = Blueprint('file', __name__)
logger = logging.getLogger(__name__)

# Handle file serving routes
@file_bp.route("/uploads/<path:filename>", methods=['OPTIONS'])
def handle_options(filename: str) -> Response:
    """Handle OPTIONS requests for CORS preflight."""
    response = Response()
    response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': '*'
    })
    return response

@file_bp.route("/uploads/<path:filename>", methods=["GET"])
def serve_file(filename: str) -> Response:
    """
    Serve a file from local or remote storage.
    
    Args:
        filename: The name/path of the file to serve
        
    Returns:
        Flask Response containing the file or error message
    """
    logger.info(f"Attempting to serve file: {filename}")
    
    range_header = request.headers.get('Range')
    
    try:
        # Check possible local file locations
        possible_paths = [
            os.path.join(UPLOADS_DIR, filename),  # Uploads directory
            os.path.join(os.getcwd(), 'uploads', filename),  # Direct uploads directory
            os.path.join(os.getcwd(), 'temp', filename),  # Local temp directory
            os.path.join(tempfile.gettempdir(), filename)  # System temp directory
        ]
        
        # Look for file in local paths
        local_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                local_path = path
                logger.info(f"found file locally at: {local_path}")
                break
                
        if local_path:
            # Get file metadata
            file_size = os.path.getsize(local_path)
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or 'application/octet-stream'
            
            logger.info(f"Serving local file: {local_path} with mime type: {mime_type}")
            
            # Handle range requests
            if range_header:
                start, end = 0, file_size - 1
                
                if range_header.startswith('bytes='):
                    ranges = range_header[6:].split('-')
                    if ranges[0]:
                        start = int(ranges[0])
                    if len(ranges) > 1 and ranges[1]:
                        end = min(int(ranges[1]), file_size - 1)
                
                with open(local_path, 'rb') as f:
                    f.seek(start)
                    data = f.read(end - start + 1)
                
                # Return partial content response
                return Response(
                    data,
                    206,
                    mimetype=mime_type,
                    direct_passthrough=True,
                    headers={
                        'Content-Range': f'bytes {start}-{end}/{file_size}',
                        'Accept-Ranges': 'bytes',
                        'Content-Length': str(end - start + 1),
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, OPTIONS',
                        'Access-Control-Allow-Headers': '*'
                    }
                )
            
            # Return full file
            return send_file(local_path, mimetype=mime_type)
        
        # Try remote storage if available
        if STORAGE_TYPE != "local" and hasattr(storage, 'get_file_bytes'):
            logger.info(f"Attempting remote storage retrieval: {filename}")
            file_bytes = storage.get_file_bytes(filename)
            
            if file_bytes:
                mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                logger.info(f"Serving remote file: {filename} with mime type: {mime_type}")
                
                return Response(
                    file_bytes,
                    mimetype=mime_type,
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, OPTIONS',
                        'Access-Control-Allow-Headers': '*'
                    }
                )
        else:
            logger.error("StorageManager missing get_file_bytes method")
        
        logger.error(f"File not found: {filename}")
        return "File not found", 404
            
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

# 
@file_bp.route("/api/recordings", methods=['POST'])
def upload_recording() -> Response:
    """
    Handle recording uploads with automatic audio extraction.
    
    Expects a video file in the request and extracts audio if present.
    Saves both video and audio files using the storage manager.
    
    Returns:
        JSON response with file URLs and metadata
    """
    try:
        logger.info(f"Received request to /api/recordings with method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request files: {request.files.keys()}")
        logger.info(f"Request form data: {request.form.keys()}")
        
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "Video file is required"}), 400
            
        video_file = request.files['video']
        logger.info(f"video info: {video_file.filename}, content type: {video_file.content_type}, size: {video_file.content_length}")
        
        if not video_file.filename:
            return jsonify({"error": "No selected file"}), 400
        
        # Extract and validate device ID
        try:
            device_id_hash = video_file.filename.split("_")[0]
            if not device_id_hash or len(device_id_hash) < 6:
                return jsonify({"error": "Invalid device ID in filename"}), 400
        except Exception as e:
            logger.error(f"Error extracting device ID hash: {e}")
            return jsonify({"error": f"Could not extract device ID hash from filename: {video_filename}"}), 400
        
        # Save video file
        video_path = storage.save_file(video_file, video_file.filename, content_type='video/mp4')
        
        # Set up temp processing
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_video = os.path.join(temp_dir, video_file.filename)
        
        # Save temp copy
        try:
            video_file.seek(0)
            video_file.save(temp_video)
            if not os.path.exists(temp_video):
                raise IOError(f"Failed to save temp video: {temp_video}")
        except Exception as e:
            logger.error(f"Error saving to temp: {str(e)}")
            return jsonify({"error": f"Failed to save temporary video file: {str(e)}"}), 500
        
        # Check for audio stream
        has_audio = False
        audio_filename = None
        
        probe_result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'json',
            temp_video
        ], capture_output=True, text=True)
        
        logger.info(f"FFprobe stdout: {probe_result.stdout}")
        logger.info(f"FFprobe stderr: {probe_result.stderr}")
        logger.info(f"FFprobe return code: {probe_result.returncode}")
        
        has_audio = False
        audio_filename = None
        
        try:
            probe_data = json.loads(probe_result.stdout)
            has_audio = (
                probe_result.returncode == 0 and
                probe_data.get('streams') and
                probe_data['streams'][0].get('codec_type') == 'audio'
            )
        except json.JSONDecodeError:
            logger.error("Error parsing ffprobe output")
            has_audio = False
        
        if has_audio:
            try:
                audio_filename = video_file.filename.replace('.mp4', '_audio.wav')
                temp_audio = os.path.join(temp_dir, audio_filename)
                
                # Extract audio
                ffmpeg_result = subprocess.run([
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', temp_video,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ac', '2',  # Stereo
                    '-ar', '44100',  # 44.1kHz sampling rate
                    '-hide_banner',
                    '-loglevel', 'info',
                    temp_audio
                ], capture_output=True, text=True)
                
                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                    with open(temp_audio, 'rb') as audio_file:
                        audio_content = audio_file.read()
                        audio_file_obj = BytesIO(audio_content)
                        storage.save_file(audio_file_obj, audio_filename, content_type='audio/wav')
                else:
                    has_audio = False
                    audio_filename = None
                
                # Clean up temp files
                for temp_file in [temp_audio, temp_video]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            logger.error(f"Failed to clean up temp file: {temp_file}, error: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Audio extraction error: {str(e)}")
                logger.error(traceback.format_exc())
                has_audio = False
                audio_filename = None
        
        # Clean up temporary video file if it still exists
        if os.path.exists(temp_video):
            try:
                logger.info(f"Cleaning up temporary video file: {temp_video}")
                os.remove(temp_video)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary video file: {str(e)}")
        
        # Get URLs for the files
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
        logger.error(f"Recording Upload Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to upload recording"}), 500

@file_bp.route("/api/recordings", methods=['GET'])
def get_recording() -> Response:
    """
    Retrieve a recording file by filename.
    
    Args:
        filename: Query parameter specifying the file to retrieve
        
    Returns:
        The requested file or error response
    """
    try:
        filename = request.args.get('filename')
        if not filename:
            return jsonify({"error": "Filename required"}), 400
            
        # Validate device ID
        device_id_hash = filename.split("_")[0]
        if not device_id_hash or len(device_id_hash) < 6:
            return jsonify({"error": "Invalid device ID"}), 400
        
        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        content_type = ('video/mp4' if filename.endswith('.mp4') else
                       'audio/wav' if filename.endswith('.wav') else
                       'application/octet-stream')
        
        return send_file(
            file_path,
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Recording Retrieval Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Failed to retrieve recording"}), 500

@file_bp.route("/api/recordings", methods=['OPTIONS'])
def handle_recordings_options() -> Response:
    """Handle CORS preflight for /api/recordings endpoint."""
    response = Response()
    response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Credentials': 'true'
    })
    return response

def cleanup_old_temp_files() -> None:
    """Delete temporary files older than 1 hour."""
    deleted_count = 0
    temp_dirs = [tempfile.gettempdir(), os.path.join(os.getcwd(), 'temp')]
    
    for temp_dir in temp_dirs:
        try:
            if not os.path.exists(temp_dir):
                logger.info(f"Directory does not exist, skipping cleanup: {temp_dir}")
                continue
                
            logger.info(f"Cleaning up temporary files in: {temp_dir}")
            
            # Find all MP4 and WAV files in temp directory
            mp4_files = []
            wav_files = []
            
            try:
                for filename in os.listdir(temp_dir):
                    if filename.endswith('.mp4'):
                        mp4_files.append(os.path.join(temp_dir, filename))
                    elif filename.endswith('.wav') or filename.endswith('_audio.wav'):
                        wav_files.append(os.path.join(temp_dir, filename))
            except Exception as e:
                logger.error(f"Error listing files in {temp_dir}: {str(e)}")
                continue
            
            logger.info(f"Found {len(mp4_files)} MP4 files and {len(wav_files)} WAV files in {temp_dir}")
            
            # Check file age and delete if older than 1 hour
            current_time = datetime.datetime.now()
            deleted_count = 0
            
            for file_path in mp4_files + wav_files:
                try:
                    # Get file creation time or modification time
                    file_mtime = os.path.getmtime(file_path)
                    file_time = datetime.datetime.fromtimestamp(file_mtime)
                    age = (current_time - file_time).total_seconds() / 3600  # Age in hours
                    
                    # If file is older than 1 hour, delete it
                    if age > 1:
                        try:
                            logger.info(f"Deleting old temporary file: {file_path} (age: {age:.2f} hours)")
                            os.remove(file_path)
                            deleted_count += 1
                            overall_deleted_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting file {file_path}: {str(e)}")
                            # Continue with other files
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    # Continue with other files
            
            logger.info(f"Cleanup complete for {temp_dir}. Deleted {deleted_count} old temporary files.")
        except Exception as e:
            logger.error(f"Error cleaning up directory {temp_dir}: {str(e)}")
            # Continue with other directories
    
    logger.info(f"Overall cleanup complete. Total deleted: {overall_deleted_count} files.")

# Clean up temp files on startup
def cleanup_on_startup():
    """Delete all video and audio files from temp directories on startup"""
    overall_deleted_count = 0
    
    # Check both system temp and application temp directories, and also uploads directories
    directories_to_clean = [
        tempfile.gettempdir(),                      # System temp directory
        os.path.join(os.getcwd(), 'temp'),          # Application temp directory
        os.path.join(os.getcwd(), 'uploads'),       # Uploads directory
        os.path.join(os.getcwd(), 'public', 'uploads')  # Public uploads directory
    ]
    
    for directory in directories_to_clean:
        try:
            if not os.path.exists(directory):
                logger.info(f"Directory does not exist, skipping cleanup: {directory}")
                continue
                
            logger.info(f"Cleaning up all media files on startup in: {directory}")
            
            # Find all MP4 and WAV files in the directory
            media_files = []
            
            try:
                for filename in os.listdir(directory):
                    if filename.endswith(('.mp4', '.wav', '_audio.wav')):
                        file_path = os.path.join(directory, filename)
                        if os.path.isfile(file_path):
                            media_files.append(file_path)
            except Exception as e:
                logger.error(f"Error listing files in {directory}: {str(e)}")
                continue
            
            logger.info(f"Found {len(media_files)} media files in {directory}")
            
            # Delete all found files
            deleted_count = 0
            for file_path in media_files:
                try:
                    logger.info(f"Deleting media file on startup: {file_path}")
                    os.remove(file_path)
                    deleted_count += 1
                    overall_deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")
                    # Continue with other files
            
            logger.info(f"Startup cleanup complete for {directory}. Deleted {deleted_count} media files.")
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory}: {str(e)}")
    
    logger.info(f"Startup cleanup complete. Total deleted: {deleted_count} files.")

# Run initial cleanup
logger.info("Running startup cleanup...")
cleanup_on_startup()

# Set up hourly cleanup
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_temp_files, 'interval', hours=1)
scheduler.start()

# Register shutdown handler
atexit.register(lambda: scheduler.shutdown(wait=False))