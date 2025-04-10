"""
Emotion detection service using DeepFace.

This module provides functionality for detecting emotions in images using the DeepFace library.
It handles graceful fallback when DeepFace is not available and provides detailed emotion analysis.
"""
import os
import numpy as np
from typing import Dict, List, Any, Optional
import traceback
from api.app import logger

# Attempt to import DeepFace, gracefully handle if unavailable
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logger.warning("DeepFace not available - emotion detection will be disabled")
    DEEPFACE_AVAILABLE = False

def detect_emotions(image_array: np.ndarray) -> Dict[str, Any]:
    """
    Detect and analyse emotions in an image using DeepFace.
    
    Args:
        image_array: A NumPy array containing the image data to analyse
        
    Returns:
        Dict containing:
            - success (bool): Whether analysis succeeded
            - emotions (Dict): Detected emotion probabilities if successful
            - dominant_emotion (str): Most prominent emotion if successful
            - error (str): Error message if analysis failed
            
    Raises:
        None.
    """
    # Check if DeepFace is available before proceeding
    if not DEEPFACE_AVAILABLE:
        return {"success": False, "error": "DeepFace library not available"}
    
    try:
        # Analyse the face
        analysis = DeepFace.analyze(
            img_path=image_array,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # Handle case where multiple faces are detected
        if isinstance(analysis, list):
            # TODO: Consider how best to handle multiple faces
            analysis = analysis[0]  # Currently using first face only
            
        # Extract relevant emotion data from analysis
        emotions: Dict[str, float] = analysis.get('emotion', {})
        dominant_emotion: str = analysis.get('dominant_emotion', 'unknown')
        
        return {
            "success": True,
            "emotions": emotions,
            "dominant_emotion": dominant_emotion
        }
        
    except Exception as e:
        # Log error details for debugging
        logger.error(f"Error during emotion detection: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e)
        }