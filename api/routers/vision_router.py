"""Router for vision-related endpoints, including face detection and gaze tracking.

Processes images and video streams to detect facial features, gaze direction,
and emotions using MediaPipe for face detection and DeepFace for emotion analysis.
"""
import base64
import io
import logging
import os

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, HTTPException
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from pydantic import BaseModel

vision_router = APIRouter()

emotion_detector = None
try:
    from deepface import DeepFace
    emotion_detector = DeepFace
    logging.info("✓ Initialised emotion detector (DeepFace)")
except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Emotion detection disabled: %s", e)

try:
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "face_landmarker_v2_with_blendshapes.task"
    )
    logging.info("Looking for face landmarker at: %s", model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    logging.info("✓ Initialised MediaPipe Face Detector")
except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Failed to initialise MediaPipe Face Detector: %s", e)
    detector = None


class ImageBody(BaseModel):
    """Request body containing a base64-encoded image."""

    image: str


def determine_gaze_direction(face_landmarks) -> str:
    """Determine gaze direction from face landmarks using iris positions.

    Uses iris positions relative to eye corners to infer where the subject
    is looking.

    Args:
        face_landmarks: MediaPipe face landmarks

    Returns:
        Gaze direction string (eg: 'center', 'up', 'down-left')
    """
    left_iris_indices = list(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
    right_iris_indices = list(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)

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

    left_eye_outer = face_landmarks[33]
    left_eye_inner = face_landmarks[133]
    right_eye_outer = face_landmarks[263]
    right_eye_inner = face_landmarks[362]

    left_eye_center = np.mean([
        [left_eye_outer.x, left_eye_outer.y],
        [left_eye_inner.x, left_eye_inner.y]
    ], axis=0)
    right_eye_center = np.mean([
        [right_eye_outer.x, right_eye_outer.y],
        [right_eye_inner.x, right_eye_inner.y]
    ], axis=0)

    x_diff = (left_iris[0] + right_iris[0]) / 2 - 0.5

    left_y_diff = left_iris[1] - left_eye_center[1]
    right_y_diff = right_iris[1] - right_eye_center[1]
    y_diff = (left_y_diff + right_y_diff) / 2

    x_threshold = 0.05
    y_threshold = 0.02

    if abs(x_diff) < x_threshold and abs(y_diff) < y_threshold:
        return "center"

    vertical = ""
    horizontal = ""

    if y_diff < -y_threshold:
        vertical = "up"
    elif y_diff > y_threshold:
        vertical = "down"

    if x_diff < -x_threshold:
        horizontal = "left"
    elif x_diff > x_threshold:
        horizontal = "right"

    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    return vertical or horizontal or "center"


def process_frame(frame) -> np.ndarray:
    """Process a video frame to detect and visualise face landmarks.

    Args:
        frame: BGR format frame from video capture

    Returns:
        Frame with landmarks drawn, in BGR format
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    if not detection_result.face_landmarks:
        return frame

    face_landmarks = detection_result.face_landmarks[0]
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=(
            mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        )
    )

    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=(
            mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
        )
    )

    solutions.drawing_utils.draw_landmarks(
        image=rgb_frame,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=(
            mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    )

    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


def gen_frames():
    """Generate processed video frames for streaming."""
    logging.info("Starting video capture")
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            try:
                processed_frame = process_frame(frame)
                ret, buffer = cv2.imencode(".jpg", processed_frame)
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Error processing frame: %s", e, exc_info=True)
                ret, buffer = cv2.imencode(".jpg", frame)
                frame = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
    finally:
        camera.release()


@vision_router.post("/api/detect-gaze")
async def detect_gaze(body: ImageBody) -> dict:
    """Detect gaze direction in an uploaded image.

    Args:
        body: JSON with base64-encoded image in 'image' field

    Returns:
        Dict with gaze direction and face landmarks

    Raises:
        HTTPException: On decoding or processing failure
    """
    try:
        image_data = body.image
        image_data = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)

        detection_result = detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return {"success": True, "face_detected": False}

        face_landmarks = detection_result.face_landmarks[0]
        gaze_direction = determine_gaze_direction(face_landmarks)
        landmarks = [[lm.x, lm.y] for lm in face_landmarks]

        x_coords = [lm.x for lm in face_landmarks]
        y_coords = [lm.y for lm in face_landmarks]
        face_box = {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(x_coords) - min(x_coords),
            "height": max(y_coords) - min(y_coords)
        }

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

        return {
            "success": True,
            "face_detected": True,
            "gaze_direction": gaze_direction,
            "landmarks": landmarks,
            "face_box": face_box,
            "gaze_arrow": gaze_arrow
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error in gaze detection endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@vision_router.post("/api/detect-combined")
async def detect_combined(body: ImageBody) -> dict:
    """Detect both gaze direction and emotions in an uploaded image.

    Args:
        body: JSON with base64-encoded image in 'image' field

    Returns:
        Dict with combined gaze and emotion detection results

    Raises:
        HTTPException: On decoding or processing failure
    """
    try:
        image_data = body.image
        # Handle data URL format (eg: "data:image/jpeg;base64,/9j/4AAQSkZJRg...")
        image_data = image_data.split(",")[1] if "," in image_data else image_data

        padding = 4 - (len(image_data) % 4)
        if padding != 4:
            image_data += "=" * padding

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as decode_error:
            logging.error("Base64 decoding error: %s", decode_error)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image data: {decode_error}"
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as image_error:
            logging.error("Image opening error: %s", image_error)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to open image: {image_error}"
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_arr = np.array(image)

        result: dict = {"success": True, "face_detected": False}

        if emotion_detector:
            try:
                image_bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
                analysis = emotion_detector.analyze(
                    img_path=image_bgr,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotions = analysis.get("emotion", {})
                box = analysis.get("region", {})

                if emotions:
                    result["face_detected"] = True
                    result["emotions"] = emotions
                    height, width = image_arr.shape[:2]
                    result["face_box"] = {
                        "x": box.get("x", 0) / width,
                        "y": box.get("y", 0) / height,
                        "width": box.get("w", 0) / width,
                        "height": box.get("h", 0) / height
                    }
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Error in emotion detection: %s", e)

        if detector:
            try:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
                detection_result = detector.detect(mp_image)

                if detection_result.face_landmarks:
                    result["face_detected"] = True
                    face_landmarks = detection_result.face_landmarks[0]
                    gaze_direction = determine_gaze_direction(face_landmarks)
                    landmarks = [[lm.x, lm.y] for lm in face_landmarks]

                    if "face_box" not in result:
                        x_coords = [lm.x for lm in face_landmarks]
                        y_coords = [lm.y for lm in face_landmarks]
                        result["face_box"] = {
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "width": max(x_coords) - min(x_coords),
                            "height": max(y_coords) - min(y_coords)
                        }

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
                        "start": {
                            "x": float(eye_center[0]),
                            "y": float(eye_center[1])
                        },
                        "end": {
                            "x": float(arrow_end[0]),
                            "y": float(arrow_end[1])
                        }
                    }

                    result["gaze"] = {
                        "direction": gaze_direction,
                        "landmarks": landmarks,
                        "gaze_arrow": gaze_arrow
                    }
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Error in gaze detection: %s", e)

        return result

    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Error in combined detection endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
