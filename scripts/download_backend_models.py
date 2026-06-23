#!/usr/bin/env python3
"""Download and warm up all backend models before server startup."""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

_MODELS = {
    "face_landmarker_v2_with_blendshapes.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker"
        "/face_landmarker/float16/1/face_landmarker.task"
    ),
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models"
        "/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
}


def _download_file(url: str, path: str) -> bool:
    """Download a file to disk with a progress bar.

    Args:
        url: Source URL
        path: Destination file path

    Returns:
        True if download succeeded
    """
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        file_size = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=file_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {Path(path).name}",
        )

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress.update(f.write(chunk))

        progress.close()
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error downloading {Path(path).name}: {e}")
        Path(path).unlink(missing_ok=True)
        return False


def _download_mediapipe_models(models_dir: str) -> bool:
    """Download MediaPipe and OpenCV models if not already present.

    Args:
        models_dir: Directory to save model files into

    Returns:
        True if all models present or downloaded successfully
    """
    all_ok = True
    for model_name, url in _MODELS.items():
        model_path = Path(models_dir) / model_name
        if model_path.exists():
            print(f"✓ {model_name} already present")
            continue
        model_path = str(model_path)
        print(f"Downloading {model_name}...")
        if not _download_file(url, model_path):
            all_ok = False
    return all_ok


def _warmup_supertonic() -> bool:
    """Trigger supertonic model download/cache warm-up.

    Imports the TTS service so auto_download runs before the server starts,
    avoiding a cold download on the first request.

    Returns:
        True if the model loaded successfully
    """
    print("Warming up supertonic TTS model...")
    try:
        # Import triggers module-level _tts = _SupertonicTTS(auto_download=True)
        from api.services.tts_service import _tts  # pylint: disable=import-outside-toplevel

        if _tts is not None:
            print("✓ Supertonic TTS model ready")
            return True
        print("✗ Supertonic TTS model failed to load")
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"✗ Could not warm up supertonic: {e}")
        return False


def main() -> int:
    """Download all required models and warm up TTS before server start."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "api" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    mediapipe_ok = _download_mediapipe_models(str(models_dir))
    supertonic_ok = _warmup_supertonic()

    if mediapipe_ok and supertonic_ok:
        print("✓ All models ready.")
        return 0

    print("✗ Some models failed — check output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
