"""Transcription service using Whisper.

Provides audio transcription via OpenAI Whisper with multiple fallback
strategies to handle edge cases such as empty tensors or KV-cache errors.
"""
import logging
import os

import soundfile as sf
import torch
import whisper

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
            logging.info("✓ Whisper model loaded on %s", device)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Whisper loading can fail for many reasons (missing model, CUDA errors)
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
        logging.info("Starting transcription of: %s", audio_path)

        model = get_whisper_model()
        if model is None:
            raise RuntimeError("Transcription model not initialised")

        try:
            audio_info = sf.info(audio_path)
            if audio_info.frames == 0:
                raise RuntimeError("Empty or invalid audio file")
            logging.info(
                "Audio file: %d frames at %d Hz",
                audio_info.frames,
                audio_info.samplerate,
            )
        except Exception as audio_error:  # pylint: disable=broad-exception-caught
            # sf.info may fail on some encodings; Whisper has its own loader
            logging.warning("Could not inspect audio file: %s", audio_error)

        methods_to_try = [
            {"name": "default", "use_fp16": True, "use_kv_cache": True},
            {"name": "no_kv_cache", "use_fp16": True, "use_kv_cache": False},
            {"name": "no_fp16", "use_fp16": False, "use_kv_cache": True},
            {"name": "basic", "use_fp16": False, "use_kv_cache": False},
        ]

        last_error = None
        for method in methods_to_try:
            try:
                logging.info("Trying transcription method: %s", method["name"])

                from whisper.audio import load_audio, pad_or_trim

                audio = load_audio(audio_path)
                audio = pad_or_trim(audio)

                if len(audio) == 0 or audio.max() < 0.01:
                    logging.warning("Audio appears silent or very quiet")

                if method["use_kv_cache"]:
                    result = model.transcribe(audio_path, fp16=method["use_fp16"])
                else:
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    detected_lang = max(probs, key=probs.get)
                    logging.info("Detected language: %s", detected_lang)

                    decode_options = whisper.DecodingOptions(
                        language=detected_lang,
                        fp16=method["use_fp16"],
                        without_timestamps=True,
                    )
                    result = whisper.decode(model, mel, decode_options)
                    result = {"text": result.text if hasattr(result, "text") else ""}

                if result and result.get("text"):
                    transcribed_text = result["text"].strip()
                    if transcribed_text:
                        logging.info(
                            "Transcription successful via %s method (%d chars)",
                            method["name"],
                            len(transcribed_text),
                        )
                        return transcribed_text
                    else:
                        logging.warning(
                            "Empty transcription result with %s method",
                            method["name"],
                        )
                else:
                    logging.warning(
                        "No text result with %s method", method["name"]
                    )
            except RuntimeError as e:
                if "cannot reshape tensor of 0 elements" in str(e):
                    logging.warning(
                        "Empty tensor error with %s method, trying next",
                        method["name"],
                    )
                    last_error = e
                elif "KeyError: Linear" in str(e):
                    logging.warning(
                        "KV cache error with %s method, trying next",
                        method["name"],
                    )
                    last_error = e
                else:
                    raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("Error with %s method: %s", method["name"], e)
                last_error = e

        if last_error:
            logging.error(
                "All transcription methods failed. Last error: %s", last_error
            )
            raise last_error

        raise RuntimeError("No transcription text generated")

    except Exception as e:
        logging.error("Error in transcription: %s", e, exc_info=True)
        raise
