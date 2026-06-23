"""TTS service using the supertonic-3 model for local speech synthesis."""

import io
import logging

import soundfile as sf

DEFAULT_VOICE = "M1"
DEFAULT_SPEED = 1.05
DEFAULT_STEPS = 8
_SPEED_MIN = 0.7
_SPEED_MAX = 2.0
_TEXT_WARN_LENGTH = 5000

try:
    from supertonic import TTS as _SupertonicTTS

    _tts = _SupertonicTTS(auto_download=True)
    logging.info("✓ Supertonic TTS model loaded")
except Exception as _e:
    _tts = None
    logging.error("Failed to load supertonic TTS model: %s", _e)


class TTSError(Exception):
    """Raised when TTS generation fails.

    Attributes:
        status_code: Suggested HTTP status code for the error
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _clamp_speed(speed: float) -> float:
    try:
        speed = float(speed)
    except (ValueError, TypeError):
        return DEFAULT_SPEED
    if not _SPEED_MIN <= speed <= _SPEED_MAX:
        logging.warning("Speed %.2f outside valid range, clamping", speed)
        return max(_SPEED_MIN, min(_SPEED_MAX, speed))
    return speed


def synthesise(
    text: str,
    voice: str | None = None,
    speed: float = DEFAULT_SPEED,
) -> io.BytesIO:
    """Generate speech audio from text using the local Supertonic model.

    Args:
        text: Text to synthesise
        voice: Voice style name, eg: "M1", "F2" (default: DEFAULT_VOICE)
        speed: Speech speed, 0.7--2.0 (default: 1.05)

    Returns:
        BytesIO buffer containing WAV audio, seeked to position 0

    Raises:
        TTSError: If generation fails
    """
    if _tts is None:
        raise TTSError("TTS model not available", 500)

    if not text or not text.strip():
        raise TTSError("Text cannot be empty", 400)

    if len(text) > _TEXT_WARN_LENGTH:
        logging.warning("Long text submitted for TTS: %d chars", len(text))

    speed = _clamp_speed(speed)
    voice_name = voice or DEFAULT_VOICE

    try:
        style = _tts.get_voice_style(voice_name=voice_name)
    except Exception as e:
        raise TTSError(f"Unknown voice '{voice_name}': {e}", 400) from e

    try:
        logging.info(
            "Synthesising %d chars, voice=%s, speed=%.2f", len(text), voice_name, speed
        )
        wav, _ = _tts.synthesize(
            text=text,
            voice_style=style,
            total_steps=DEFAULT_STEPS,
            speed=speed,
            lang="en",
        )

        buffer = io.BytesIO()
        sf.write(buffer, wav.squeeze(), _tts.sample_rate, format="WAV")
        buffer.seek(0)

        if not buffer.getbuffer().nbytes:
            raise TTSError("Empty audio returned by TTS model", 500)

        return buffer

    except TTSError:
        raise
    except Exception as e:
        raise TTSError(str(e), 500) from e
