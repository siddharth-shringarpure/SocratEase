from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pyneuphonic import Neuphonic, save_audio
import io
from dotenv import load_dotenv
import traceback
import numpy as np
import base64
from PIL import Image
from deepface import DeepFace

# Debug information about Python environment
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("PYTHONPATH:", sys.path)
print("Current working directory:", os.getcwd())
print("Environment variables:", dict(os.environ))

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def detect_emotions(image: Image) -> dict:
    """
    Detects faces and emotions in an image using DeepFace.
    Returns normalized emotion scores and cropped face image.
    """
    # Convert PIL Image to numpy array for DeepFace
    image_arr = np.array(image)
    
    try:
        # Process image with DeepFace
        result = DeepFace.analyze(image_arr, 
                                actions=['emotion'],
                                enforce_detection=False)
        
        # Handle multiple faces by taking first one
        if isinstance(result, list):
            result = result[0]
        
        # Extract face region coordinates
        face_region = result.get('region', {})
        x, y = face_region.get('x', 0), face_region.get('y', 0)
        w, h = face_region.get('w', 0), face_region.get('h', 0)
        
        # Crop the face if detected
        if w > 0 and h > 0:
            # crop and encode detected face
            face = image.crop((x, y, x + w, y + h))
            
            # Convert face to base64
            face_bytes = io.BytesIO()
            face.save(face_bytes, format='JPEG')
            face_base64 = base64.b64encode(face_bytes.getvalue()).decode('utf-8')
            
            # normalize emotion scores to 0-1 range
            emotions = result['emotion']
            total = sum(emotions.values())
            normalized_emotions = {
                emotion: round(score / total, 3)
                for emotion, score in emotions.items()
            }
            
            # Sort by intensity (highest first)
            normalized_emotions = dict(
                sorted(normalized_emotions.items(), 
                      key=lambda x: x[1], 
                      reverse=True)
            )
            
            return {
                "success": True,
                "face_detected": True,
                "face_image": face_base64,
                "emotions": normalized_emotions
            }
    
    except Exception as e:
        print(f"Error detecting emotions: {str(e)}")
        traceback.print_exc()
    
    return {
        "success": True,
        "face_detected": False,
        "emotions": {}
    }

@app.route("/api/detect-emotion", methods=['POST'])
def detect_emotion():
    """Endpoint to detect emotions in uploaded images"""
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Parse base64 image data
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process the image
        result = detect_emotions(image)
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Initialize Neuphonic client with API key from environment
api_key = os.environ.get('NEUPHONIC_API_KEY')
if not api_key:
    raise ValueError("NEUPHONIC_API_KEY not found in environment variables")
client = Neuphonic(api_key=api_key)

@app.route("/api/python")
def hello_world():
    """Simple health check endpoint"""
    return "Hello, World!"

@app.route("/api/tts", methods=['POST'])
def text_to_speech():
    """Converts text to speech using Neuphonic API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        speed = max(0.7, min(2.0, float(data.get('speed', 1.0))))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"Processing TTS request - text: {text}, voice: {voice}, speed: {speed}")

        # Generate audio using Neuphonic
        sse = client.tts.SSEClient()
        sse.speed = speed
        response = sse.send(text)
        
        # Save audio to temporary buffer
        temp_file = io.BytesIO()
        save_audio(response, temp_file)
        
        # Debug audio generation
        temp_file.seek(0)
        audio_data = temp_file.read()
        print(f"Generated audio size: {len(audio_data)} bytes")
        
        temp_file.seek(0)
        
        # Return audio file with CORS headers
        response = send_file(
            temp_file,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='speech.wav'
        )
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response
    
    except Exception as e:
        print("TTS Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/test", methods=['GET'])
def test_endpoint():
    """Health check endpoint for emotion detection API"""
    return jsonify({"status": "ok", "message": "Emotion detection API is running"})

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, host='0.0.0.0', port=port)