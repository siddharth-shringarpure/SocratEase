"""
Routes for vision-related endpoints, including face detection and gaze tracking.

This module provides Flask routes for processing images and video streams to detect
facial features, gaze direction, and emotions. It uses MediaPipe for face detection
and landmark tracking, and FER for emotion detection.
"""

from flask import Blueprint, request, jsonify
import cv2
import os
import io
import base64
import numpy as np
import traceback
import logging
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# Create blueprint and logger
vision_bp = Blueprint('vision', __name__)
logger = logging.getLogger(__name__)

emotion_detector = None
try:
    # Handle moviepy dependency for FER
    import sys
    import types
    
    # Create dummy moviepy module if needed
    if 'moviepy' not in sys.modules:
        sys.modules['moviepy'] = types.ModuleType('moviepy')
        sys.modules['moviepy.editor'] = types.ModuleType('moviepy.editor')
        logger.info("Created dummy moviepy module for FER import")
    
    # Try importing and initialising FER with minimal parameters
    from fer import FER
    try:
        emotion_detector = FER()  # Most compatible approach
    except TypeError:
        try:
            emotion_detector = FER(mtcnn=False)
        except:
            # Last resort, try with no face detector
            emotion_detector = FER(face_detector=None)
    
    logger.info("Successfully initialised emotion detector")
except Exception as e:
    logger.warning(f"Emotion detection disabled: {str(e)}")

# Initialise MediaPipe Face Detector
try:
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 
        'face_landmarker_v2_with_blendshapes.task'
    )
    logger.info(f"Looking for face landmarker at: {model_path}")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    logger.info("Successfully initialised MediaPipe Face Detector")
except Exception as e:
    logger.error(f"Failed to initialise MediaPipe Face Detector: {e}")
    detector = None

def determine_gaze_direction(face_landmarks):
    """
    Determine gaze direction based on face landmarks, using iris positions
    relative to eye corners.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        
    Returns:
        str: Gaze direction (e.g., 'center', 'up', 'down-left', etc.)
    """
    # Convert iris indices to list format
    left_iris_indices = list(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
    right_iris_indices = list(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)
    
    # Calculate mean iris positions
    left_points = []
    for idx in left_iris_indices:
        if isinstance(idx, tuple):
            idx = idx[0]
        left_points.append([
            face_landmarks[idx].x,
            face_landmarks[idx].y,
            face_landmarks[idx].z
        ])
    left_iris = np.mean(left_points, axis=0)
    
    right_points = []
    for idx in right_iris_indices:
        if isinstance(idx, tuple):
            idx = idx[0]
        right_points.append([
            face_landmarks[idx].x,
            face_landmarks[idx].y,
            face_landmarks[idx].z
        ])
    right_iris = np.mean(right_points, axis=0)
    
    # Get eye corner landmarks
    left_eye_outer = face_landmarks[33]  # Outer corner of left eye
    left_eye_inner = face_landmarks[133]  # Inner corner of left eye
    right_eye_outer = face_landmarks[263]  # Outer corner of right eye
    right_eye_inner = face_landmarks[362]  # Inner corner of right eye
    
    # Calculate eye centres
    left_eye_center = np.mean([
        [left_eye_outer.x, left_eye_outer.y],
        [left_eye_inner.x, left_eye_inner.y]
    ], axis=0)
    right_eye_center = np.mean([
        [right_eye_outer.x, right_eye_outer.y],
        [right_eye_inner.x, right_eye_inner.y]
    ], axis=0)
    
    # Calculate relative positions
    x_diff = (left_iris[0] + right_iris[0]) / 2 - 0.5
    
    # Calculate vertical gaze using distance from iris to eye center
    left_y_diff = left_iris[1] - left_eye_center[1]
    right_y_diff = right_iris[1] - right_eye_center[1]
    y_diff = (left_y_diff + right_y_diff) / 2
    
    # Define thresholds
    x_threshold = 0.05  # Horizontal movement threshold
    y_threshold = 0.02  # Vertical movement threshold (more sensitive)
    
    # Return early if gaze is centered
    if abs(x_diff) < x_threshold and abs(y_diff) < y_threshold:
        return "center"
    
    vertical = ""
    horizontal = ""
    
    # Determine vertical direction
    if y_diff < -y_threshold:
        vertical = "up"
    elif y_diff > y_threshold:
        vertical = "down"
        
    # Determine horizontal direction
    if x_diff < -x_threshold:
        horizontal = "left"
    elif x_diff > x_threshold:
        horizontal = "right"
        
    # Combine directions if both present
    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "center"

def process_frame(frame):
    """
    Process a video frame to detect and visualise face landmarks.
    
    Args:
        frame: BGR format frame from video capture
        
    Returns:
        processed_frame: Frame with landmarks drawn
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect face landmarks
    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return frame
    
    face_landmarks = detection_result.face_landmarks[0]
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
        for landmark in face_landmarks
    ])

    # Draw face mesh tesselations
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style()
    )

    # Draw face mesh contours
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style()
    )

    # Draw irises
    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style()
    )
    
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

def gen_frames():
    """Generate processed video frames for streaming."""
    logger.info("Starting video capture")
    camera = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            try:
                processed_frame = process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                # Fall back to original frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

@vision_bp.route("/api/detect-gaze", methods=['POST'])
def detect_gaze():
    """
    Endpoint to detect gaze direction in uploaded images.
    
    Expects:
        JSON with base64 encoded image in 'image' field
        
    Returns:
        JSON with gaze direction and face landmarks
    """
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse and decode image
        image_data = request.json['image']
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        
        # Detect face landmarks
        detection_result = detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return jsonify({
                "success": True,
                "face_detected": False
            })
        
        face_landmarks = detection_result.face_landmarks[0]
        gaze_direction = determine_gaze_direction(face_landmarks)
        landmarks = [[landmark.x, landmark.y] for landmark in face_landmarks]
        
        # Calculate face bounding box
        x_coords = [landmark.x for landmark in face_landmarks]
        y_coords = [landmark.y for landmark in face_landmarks]
        face_box = {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }
        
        # Calculate gaze arrow
        left_eye_center = np.mean([
            [face_landmarks[33].x, face_landmarks[33].y],  # Outer corner
            [face_landmarks[133].x, face_landmarks[133].y]  # Inner corner
        ], axis=0)
        right_eye_center = np.mean([
            [face_landmarks[263].x, face_landmarks[263].y],  # Outer corner
            [face_landmarks[362].x, face_landmarks[362].y]  # Inner corner
        ], axis=0)
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
        
        arrow_length = 0.1
        arrow_end = eye_center.copy()
        
        if "left" in gaze_direction:
            arrow_end[0] -= arrow_length
        elif "right" in gaze_direction:
            arrow_end[0] += arrow_length
        if "up" in gaze_direction:
            arrow_end[1] -= arrow_length
        elif "down" in gaze_direction:
            arrow_end[1] += arrow_length
            
        gaze_arrow = {
            "start": {"x": float(eye_center[0]), "y": float(eye_center[1])},
            "end": {"x": float(arrow_end[0]), "y": float(arrow_end[1])}
        }
        
        return jsonify({
            "success": True,
            "face_detected": True,
            "gaze_direction": gaze_direction,
            "landmarks": landmarks,
            "face_box": face_box,
            "gaze_arrow": gaze_arrow
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@vision_bp.route("/api/detect-combined", methods=['POST'])
def detect_combined():
    """
    Endpoint to detect both gaze direction and emotions in uploaded images.
    
    Expects:
        JSON with base64 encoded image in 'image' field
        
    Returns:
        JSON with combined gaze and emotion detection results
    """
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse and decode image
        image_data = request.json['image']
        # Handle data URL format (eg: "data:image/jpeg;base64,/9j/4AAQSkZJRg...")
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        
        # Add padding if needed
        padding = 4 - (len(image_data) % 4)
        if padding != 4:
            image_data += '=' * padding
            
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as decode_error:
            logger.error(f"Base64 decoding error: {str(decode_error)}")
            return jsonify({
                "success": False,
                "error": f"Failed to decode image data: {str(decode_error)}"
            }), 400
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as image_error:
            logger.error(f"Image opening error: {str(image_error)}")
            return jsonify({
                "success": False,
                "error": f"Failed to open image: {str(image_error)}"
            }), 400
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_arr = np.array(image)
        
        result = {
            "success": True,
            "face_detected": False
        }
        
        # Detect emotions using FER if available
        if emotion_detector:
            try:
                image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
                emotions = emotion_detector.detect_emotions(image_bgr)
                
                if emotions and len(emotions) > 0:
                    result["face_detected"] = True
                    result["emotions"] = emotions[0]["emotions"]
                    
                    # Normalise box coordinates
                    box = emotions[0]["box"]
                    height, width = image_arr.shape[:2]
                    result["face_box"] = {
                        "x": box[0] / width,
                        "y": box[1] / height,
                        "width": box[2] / width,
                        "height": box[3] / height
                    }
            except Exception as e:
                logger.error(f"Error in emotion detection: {e}")
        
        # Detect gaze using MediaPipe if available
        if detector:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
                detection_result = detector.detect(mp_image)
                
                if detection_result.face_landmarks:
                    result["face_detected"] = True
                    face_landmarks = detection_result.face_landmarks[0]
                    gaze_direction = determine_gaze_direction(face_landmarks)
                    landmarks = [[landmark.x, landmark.y] for landmark in face_landmarks]
                    
                    # Calculate face box if not set
                    if "face_box" not in result:
                        x_coords = [landmark.x for landmark in face_landmarks]
                        y_coords = [landmark.y for landmark in face_landmarks]
                        result["face_box"] = {
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "width": max(x_coords) - min(x_coords),
                            "height": max(y_coords) - min(y_coords)
                        }
                    
                    # Calculate gaze arrow
                    left_eye_center = np.mean([
                        [face_landmarks[33].x, face_landmarks[33].y],
                        [face_landmarks[133].x, face_landmarks[133].y]
                    ], axis=0)
                    right_eye_center = np.mean([
                        [face_landmarks[263].x, face_landmarks[263].y],
                        [face_landmarks[362].x, face_landmarks[362].y]
                    ], axis=0)
                    eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
                    
                    arrow_length = 0.1
                    arrow_end = eye_center.copy()
                    
                    if "left" in gaze_direction:
                        arrow_end[0] -= arrow_length
                    elif "right" in gaze_direction:
                        arrow_end[0] += arrow_length
                    if "up" in gaze_direction:
                        arrow_end[1] -= arrow_length
                    elif "down" in gaze_direction:
                        arrow_end[1] += arrow_length
                        
                    gaze_arrow = {
                        "start": {"x": float(eye_center[0]), "y": float(eye_center[1])},
                        "end": {"x": float(arrow_end[0]), "y": float(arrow_end[1])}
                    }
                    
                    result["gaze"] = {
                        "direction": gaze_direction,
                        "landmarks": landmarks,
                        "gaze_arrow": gaze_arrow
                    }
            except Exception as e:
                logger.error(f"Error in gaze detection: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500