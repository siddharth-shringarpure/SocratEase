"""Emotion detection service using DeepFace.

This module provides functionality for detecting emotions in images using the
DeepFace library. It handles graceful fallback when DeepFace is not available
and provides detailed emotion analysis.
"""
import logging
from typing import Any

import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logging.warning("DeepFace not available -- emotion detection will be disabled")
    DEEPFACE_AVAILABLE = False


def detect_emotions(image_array: np.ndarray) -> dict[str, Any]:
    """Detect and analyse emotions in an image using DeepFace.

    Args:
        image_array: NumPy array containing the image data to analyse

    Returns:
        Dict containing:
            - success (bool): Whether analysis succeeded
            - emotions (dict): Detected emotion probabilities if successful
            - dominant_emotion (str): Most prominent emotion if successful
            - error (str): Error message if analysis failed
    """
    if not DEEPFACE_AVAILABLE:
        return {"success": False, "error": "DeepFace library not available"}

    try:
        analysis = DeepFace.analyze(
            img_path=image_array,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        # TODO: Consider how best to handle multiple faces
        if isinstance(analysis, list):
            analysis = analysis[0]

        emotions: dict[str, float] = analysis.get("emotion", {})
        dominant_emotion: str = analysis.get("dominant_emotion", "unknown")

        return {
            "success": True,
            "emotions": emotions,
            "dominant_emotion": dominant_emotion
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        # DeepFace raises heterogeneous errors depending on model/backend
        logging.error("Error during emotion detection: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
