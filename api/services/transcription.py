"""
Transcription service using Whisper.

Provides audio transcription via OpenAI Whisper with multiple fallback
strategies to handle edge cases such as empty tensors or KV-cache errors.
"""
import datetime
import logging
import os

import soundfile as sf
import torch
import whisper

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model():
    """Return the Whisper model, initialising it on first call.

    Returns:
        Loaded Whisper model instance, or None if initialisation failed
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Initialising Whisper model on device: %s", device)
            _whisper_model = whisper.load_model("small").to(device)
            logging.info("Whisper model loaded successfully on %s", device)
        except Exception as e:
            logging.error("Failed to load Whisper model: %s", e, exc_info=True)
            _whisper_model = None
    return _whisper_model


def transcribe_audio(audio_path: str) -> str | None:
    """Transcribe an audio file using Whisper.

    Attempts multiple decoding strategies (fp16/fp32, with/without KV cache)
    and returns the first successful result.

    Args:
        audio_path: Path to the audio file

    Returns:
        Transcribed text if successful, None if all methods fail

    Raises:
        RuntimeError: If the audio file is empty, invalid, or the model is
            not initialised
    """
    try:
        logging.info(
            "%s: === Starting transcription ===", datetime.datetime.now()
        )
        logging.info(
            "%s: Processing audio file: %s",
            datetime.datetime.now(),
            audio_path
        )

        model = get_whisper_model()
        if model is None:
            logging.error(
                "%s: Whisper model not properly initialised",
                datetime.datetime.now()
            )
            raise RuntimeError("Transcription model not initialised")

        try:
            audio_info = sf.info(audio_path)
            if audio_info.frames == 0:
                logging.error(
                    "%s: Audio file is empty or invalid", datetime.datetime.now()
                )
                raise RuntimeError("Empty or invalid audio file")
            logging.info(
                "%s: Audio file info: %d frames, %d Hz",
                datetime.datetime.now(),
                audio_info.frames,
                audio_info.samplerate
            )
        except Exception as audio_error:
            logging.error("Error checking audio file: %s", audio_error)
            # Whisper has its own audio loading; continue regardless

        methods_to_try = [
            {"name": "default", "use_fp16": True, "use_kv_cache": True},
            {"name": "no_kv_cache", "use_fp16": True, "use_kv_cache": False},
            {"name": "no_fp16", "use_fp16": False, "use_kv_cache": True},
            {"name": "basic", "use_fp16": False, "use_kv_cache": False}
        ]

        last_error = None
        for method in methods_to_try:
            try:
                logging.info(
                    "%s: Trying transcription method: %s",
                    datetime.datetime.now(),
                    method["name"]
                )

                from whisper.audio import load_audio, pad_or_trim

                audio = load_audio(audio_path)
                audio = pad_or_trim(audio)

                if len(audio) == 0 or audio.max() < 0.01:
                    logging.warning(
                        "%s: Audio appears silent or very quiet",
                        datetime.datetime.now()
                    )

                if method["use_kv_cache"]:
                    result = model.transcribe(audio_path, fp16=method["use_fp16"])
                else:
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    detected_lang = max(probs, key=probs.get)
                    logging.info(
                        "%s: Detected language: %s",
                        datetime.datetime.now(),
                        detected_lang
                    )

                    decode_options = whisper.DecodingOptions(
                        language=detected_lang,
                        fp16=method["use_fp16"],
                        without_timestamps=True
                    )
                    result = whisper.decode(model, mel, decode_options)
                    result = {"text": result.text if hasattr(result, "text") else ""}

                if result and result.get("text"):
                    transcribed_text = result["text"].strip()
                    if transcribed_text:
                        logging.info(
                            "%s: Transcription successful with %s method",
                            datetime.datetime.now(),
                            method["name"]
                        )
                        logging.info(
                            "%s: Transcribed text length: %d characters",
                            datetime.datetime.now(),
                            len(transcribed_text)
                        )
                        return transcribed_text
                    else:
                        logging.warning(
                            "%s: Empty transcription result with %s method",
                            datetime.datetime.now(),
                            method["name"]
                        )
                else:
                    logging.warning(
                        "%s: No text result with %s method",
                        datetime.datetime.now(),
                        method["name"]
                    )
            except RuntimeError as e:
                if "cannot reshape tensor of 0 elements" in str(e):
                    logging.warning(
                        "%s: Empty tensor error with %s method, trying next",
                        datetime.datetime.now(),
                        method["name"]
                    )
                    last_error = e
                elif "KeyError: Linear" in str(e):
                    logging.warning(
                        "%s: KV cache error with %s method, trying next",
                        datetime.datetime.now(),
                        method["name"]
                    )
                    last_error = e
                else:
                    raise
            except Exception as e:
                logging.warning(
                    "%s: Error with %s method: %s",
                    datetime.datetime.now(),
                    method["name"],
                    e
                )
                last_error = e

        if last_error:
            logging.error(
                "%s: All transcription methods failed. Last error: %s",
                datetime.datetime.now(),
                last_error
            )
            raise last_error
        else:
            logging.error(
                "%s: No transcription text in result", datetime.datetime.now()
            )
            raise RuntimeError("No transcription text generated")

    except Exception as e:
        logging.error(
            "%s: Error in transcription: %s",
            datetime.datetime.now(),
            e,
            exc_info=True
        )
        raise
