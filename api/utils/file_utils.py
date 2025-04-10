"""
Utility functions for file handling and audio processing.

This module provides helper functions for:
- Validating audio file extensions
- Extracting audio from media files using FFmpeg
"""
import os
import subprocess
from typing import Set


def allowed_audio_file(filename: str) -> bool:
    """
    Check if a file has an allowed audio extension.

    Args:
        filename: Name of the file to check

    Returns:
        bool: True if extension is allowed, False otherwise
    """
    # Define allowed audio formats
    ALLOWED_EXTENSIONS: Set[str] = {
        'mp3', 'wav', 'mp4', 'm4a', 'ogg', 'webm'
    }

    # Check for extension presence and validate it
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def extract_audio(input_path: str) -> str:
    """
    Extract audio from input file using FFmpeg.

    Args:
        input_path: Path to the input media file

    Returns:
        str: Path to the extracted WAV audio file

    Raises:
        Exception: If audio extraction fails
    """
    # Generate output path with .wav extension
    output_path = os.path.splitext(input_path)[0] + '.wav'

    # Configure FFmpeg parameters for audio extraction
    ffmpeg_params = [
        'ffmpeg',
        '-i', input_path,  # Input file
        '-vn',  # Disable video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '16000',  # Sample rate
        '-ac', '1',  # Mono audio
        output_path
    ]

    try:
        # Run FFmpeg command
        subprocess.run(
            ffmpeg_params,
            check=True,
            capture_output=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        # TODO: Add more detailed error handling and logging
        raise Exception("Failed to extract audio from file") 