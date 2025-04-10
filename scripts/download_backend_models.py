#!/usr/bin/env python3

import os
import sys
import requests
from tqdm import tqdm

# Model information with URLs
MODELS = {
    'face_landmarker_v2_with_blendshapes.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'face_detection_yunet_2023mar.onnx': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
}

def download_file(url, filename):
    """
    Download a file with progress bar.
    Returns True if download was successful.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        file_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress = tqdm(
            total=file_size,
            unit='iB',
            unit_scale=True,
            desc=f'Downloading {filename}'
        )
        
        # Download file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    progress.update(size)
        
        progress.close()
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def main():
    """Main function to download all required models."""
    # Get the project root directory (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'api', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Track if all downloads were successful
    all_successful = True
    
    # Download each model
    for model_name, url in MODELS.items():
        print(f"\nProcessing {model_name}...")
        
        # Full path for the model file
        model_path = os.path.join(models_dir, model_name)
        
        # Check if file already exists
        if os.path.exists(model_path):
            print(f"{model_name} already exists")
            continue
        
        # Download the model
        if not download_file(url, model_path):
            all_successful = False
            print(f"Failed to download {model_name}")
    
    if all_successful:
        print("\nAll models downloaded successfully!")
        return 0
    else:
        print("\nSome models failed to download. Please try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 