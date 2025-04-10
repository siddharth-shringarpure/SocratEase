"""
Transcription service using Whisper
"""
import os
import whisper
import datetime
from typing import Optional
import soundfile as sf
import torch
import logging

# Create a local logger instead of importing from api.app
logger = logging.getLogger(__name__)

# We'll use a singleton pattern for the model
_whisper_model = None

def get_whisper_model():
    """Get the Whisper model, initializing it if necessary"""
    global _whisper_model
    if _whisper_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initialising Whisper model on device: {device}")
            _whisper_model = whisper.load_model("small").to(device)
            logger.info(f"Whisper model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            _whisper_model = None
    return _whisper_model

def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribes an audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        str: Transcribed text if successful 
        None: If transcription fails

    Excepts:
        RuntimeError: If the audio file is empty or invalid
        Exception: If an error occurs during transcription
    """
    try:
        logger.info(f"{datetime.datetime.now()}: === Starting transcription ===")
        logger.info(f"{datetime.datetime.now()}: Processing audio file: {audio_path}")
        
        model = get_whisper_model()
        if model is None:
            logger.error(f"{datetime.datetime.now()}: Whisper model not properly initialised")
            raise RuntimeError("Transcription model not initialised")

        # First make sure the audio file is valid
        try:
            audio_info = sf.info(audio_path)
            if audio_info.frames == 0:
                logger.error(f"{datetime.datetime.now()}: Audio file is empty or invalid")
                raise RuntimeError("Empty or invalid audio file")
            logger.info(f"{datetime.datetime.now()}: Audio file info: {audio_info.frames} frames, {audio_info.samplerate} Hz")
        except Exception as audio_error:
            logger.error(f"Error checking audio file: {audio_error}")
            # Continue anyway as whisper has its own audio loading

        # Try with different methods in sequence
        methods_to_try = [
            {"name": "default", "use_fp16": True, "use_kv_cache": True},
            {"name": "no_kv_cache", "use_fp16": True, "use_kv_cache": False},
            {"name": "no_fp16", "use_fp16": False, "use_kv_cache": True},
            {"name": "basic", "use_fp16": False, "use_kv_cache": False}
        ]
        
        last_error = None
        for method in methods_to_try:
            try:
                logger.info(f"{datetime.datetime.now()}: Trying transcription method: {method['name']}")
                
                # Use lower-level Whisper API to have more control
                import whisper
                from whisper.audio import load_audio, pad_or_trim
                from whisper.utils import get_writer
                
                # Load and prepare audio
                audio = load_audio(audio_path)
                audio = pad_or_trim(audio)
                
                # Check if audio contains data
                if len(audio) == 0 or audio.max() < 0.01:
                    logger.warning(f"{datetime.datetime.now()}: Audio seems to be silent or very quiet, may fail transcription")
                
                # Create specific decode options
                decode_options = whisper.DecodingOptions(
                    fp16=method["use_fp16"],
                    without_timestamps=True
                )
                
                if method["use_kv_cache"]:
                    # Use standard method
                    result = model.transcribe(audio_path, fp16=method["use_fp16"])
                else:
                    # Use method without KV cache
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)
                    _, probs = model.detect_language(mel)
                    detected_lang = max(probs, key=probs.get)
                    logger.info(f"{datetime.datetime.now()}: Detected language: {detected_lang}")
                    
                    decode_options = whisper.DecodingOptions(
                        language=detected_lang, 
                        fp16=method["use_fp16"],
                        without_timestamps=True
                    )
                    result = whisper.decode(model, mel, decode_options)
                    result = {"text": result.text if hasattr(result, 'text') else ""}
                
                if result and result.get("text"):
                    transcribed_text = result["text"].strip()
                    if transcribed_text:
                        logger.info(f"{datetime.datetime.now()}: Transcription successful with {method['name']} method")
                        logger.info(f"{datetime.datetime.now()}: Transcribed text length: {len(transcribed_text)} characters")
                        return transcribed_text
                    else:
                        logger.warning(f"{datetime.datetime.now()}: Empty transcription result with {method['name']} method")
                else:
                    logger.warning(f"{datetime.datetime.now()}: No text result with {method['name']} method")
            except RuntimeError as e:
                if "cannot reshape tensor of 0 elements" in str(e):
                    logger.warning(f"{datetime.datetime.now()}: Empty tensor error with {method['name']} method, trying next method")
                    last_error = e
                elif "KeyError: Linear" in str(e):
                    logger.warning(f"{datetime.datetime.now()}: KV cache error with {method['name']} method, trying next method")
                    last_error = e
                else:
                    # Unknown runtime error, raise it
                    raise
            except Exception as e:
                logger.warning(f"{datetime.datetime.now()}: Error with {method['name']} method: {e}")
                last_error = e
        
        # If we reach here, all methods failed
        if last_error:
            logger.error(f"{datetime.datetime.now()}: All transcription methods failed. Last error: {last_error}")
            raise last_error
        else:
            logger.error(f"{datetime.datetime.now()}: No transcription text in result")
            raise RuntimeError("No transcription text generated")
    except Exception as e:
        logger.error(f"{datetime.datetime.now()}: Error in transcription: {e}", exc_info=True)
        raise

# def transcribe_long_audio(audio_path: str) -> Optional[str]:
#     """
#     Transcribe audio file using Whisper model.
    
#     Args:
#         audio_path: Path to the audio file
        
#     Returns:
#         str: Transcribed text if successful 
#         None: If transcription fails
#     """
#     try:
#         logger.info(f"{datetime.datetime.now()}: === Starting transcription ===")
#         logger.info(f"{datetime.datetime.now()}: Processing audio file: {audio_path}")
        
#         if model is None:
#             logger.error(f"{datetime.datetime.now()}: Whisper model not properly initialised")
#             raise RuntimeError("Transcription model not initialised")

#         # First make sure the audio file is valid
#         try:
#             import soundfile as sf
#             audio_info = sf.info(audio_path)
#             if audio_info.frames == 0:
#                 logger.error(f"{datetime.datetime.now()}: Audio file is empty or invalid")
#                 raise RuntimeError("Empty or invalid audio file")
#             logger.info(f"{datetime.datetime.now()}: Audio file info: {audio_info.frames} frames, {audio_info.samplerate} Hz")
#         except Exception as audio_error:
#             logger.error(f"Error checking audio file: {audio_error}")
#             # Continue anyway as whisper has its own audio loading

#         # Try with different methods in sequence
#         methods_to_try = [
#             {"name": "default", "use_fp16": True, "use_kv_cache": True},
#             {"name": "no_kv_cache", "use_fp16": True, "use_kv_cache": False},
#             {"name": "no_fp16", "use_fp16": False, "use_kv_cache": True},
#             {"name": "basic", "use_fp16": False, "use_kv_cache": False}
#         ]
        
#         last_error = None
#         for method in methods_to_try:
#             try:
#                 logger.info(f"{datetime.datetime.now()}: Trying transcription method: {method['name']}")
                
#                 # Use lower-level Whisper API to have more control
#                 import whisper
#                 from whisper.audio import load_audio, pad_or_trim
#                 from whisper.utils import get_writer
                
#                 # Load and prepare audio
#                 audio = load_audio(audio_path)
#                 audio = pad_or_trim(audio)
                
#                 # Check if audio contains data
#                 if len(audio) == 0 or audio.max() < 0.01:
#                     logger.warning(f"{datetime.datetime.now()}: Audio seems to be silent or very quiet, may fail transcription")
                
#                 # Create specific decode options
#                 decode_options = whisper.DecodingOptions(
#                     fp16=method["use_fp16"],
#                     without_timestamps=True
#                 )
                
#                 if method["use_kv_cache"]:
#                     # Use standard method
#                     result = model.transcribe(audio_path, fp16=method["use_fp16"])
#                 else:
#                     # Use method without KV cache
#                     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#                     _, probs = model.detect_language(mel)
#                     detected_lang = max(probs, key=probs.get)
#                     logger.info(f"{datetime.datetime.now()}: Detected language: {detected_lang}")
                    
#                     decode_options = whisper.DecodingOptions(
#                         language=detected_lang, 
#                         fp16=method["use_fp16"],
#                         without_timestamps=True
#                     )
#                     result = whisper.decode(model, mel, decode_options)
#                     result = {"text": result.text if hasattr(result, 'text') else ""}
                
#                 if result and result.get("text"):
#                     transcribed_text = result["text"].strip()
#                     if transcribed_text:
#                         logger.info(f"{datetime.datetime.now()}: Transcription successful with {method['name']} method")
#                         logger.info(f"{datetime.datetime.now()}: Transcribed text length: {len(transcribed_text)} characters")
#                         return transcribed_text
#                     else:
#                         logger.warning(f"{datetime.datetime.now()}: Empty transcription result with {method['name']} method")
#                 else:
#                     logger.warning(f"{datetime.datetime.now()}: No text result with {method['name']} method")
#             except RuntimeError as e:
#                 if "cannot reshape tensor of 0 elements" in str(e):
#                     logger.warning(f"{datetime.datetime.now()}: Empty tensor error with {method['name']} method, trying next method")
#                     last_error = e
#                 elif "KeyError: Linear" in str(e):
#                     logger.warning(f"{datetime.datetime.now()}: KV cache error with {method['name']} method, trying next method")
#                     last_error = e
#                 else:
#                     # Unknown runtime error, raise it
#                     raise
#             except Exception as e:
#                 logger.warning(f"{datetime.datetime.now()}: Error with {method['name']} method: {e}")
#                 last_error = e
        
#         # If we reach here, all methods failed
#         if last_error:
#             logger.error(f"{datetime.datetime.now()}: All transcription methods failed. Last error: {last_error}")
#             raise last_error
#         else:
#             logger.error(f"{datetime.datetime.now()}: No transcription text in result")
#             raise RuntimeError("No transcription text generated")
#     except Exception as e:
#         logger.error(f"{datetime.datetime.now()}: Error in transcription: {e}", exc_info=True)
#         raise 