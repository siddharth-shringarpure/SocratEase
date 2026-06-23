"""
Utility functions for file handling and audio processing.

This module provides helper functions for:
- Validating audio file extensions
- Extracting audio from media files using FFmpeg
"""
import os
import subprocess

ALLOWED_AUDIO_EXTENSIONS: set[str] = {
    "mp3", "wav", "mp4", "m4a", "ogg", "webm"
}


def allowed_audio_file(filename: str) -> bool:
    """Check if a file has an allowed audio extension.

    Args:
        filename: Name of the file to check

    Returns:
        True if the extension is allowed, False otherwise
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS
    )


def extract_audio(input_path: str) -> str:
    """Extract audio from a media file using FFmpeg.

    Args:
        input_path: Path to the input media file

    Returns:
        Path to the extracted WAV audio file

    Raises:
        Exception: If audio extraction fails
    """
    output_path = os.path.splitext(input_path)[0] + ".wav"

    ffmpeg_params = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_params, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        # TODO: Add more detailed error handling and logging
        raise Exception("Failed to extract audio from file") from e
