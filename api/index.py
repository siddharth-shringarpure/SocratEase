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
import cv2  # OpenCV for face detection

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

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("Face detector loaded successfully!")

def detect_emotions(image):
    """
    A simple face detector that returns random emotion values.
    In a real app, you'd use a proper emotion detection model.
    """
    # Convert PIL Image to numpy array for OpenCV
    image_arr = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    image_arr = image_arr[:, :, ::-1].copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # For demo purposes, return random emotion values
        # In a real app, you'd use a proper emotion classifier here
        return {
            "success": True,
            "face_detected": True,
            "emotions": {
                "angry": np.random.random() * 0.2,
                "disgust": np.random.random() * 0.1,
                "fear": np.random.random() * 0.1,
                "happy": np.random.random() * 0.5,
                "sad": np.random.random() * 0.2,
                "surprise": np.random.random() * 0.2,
                "neutral": np.random.random() * 0.4
            }
        }
    
    return {
        "success": True,
        "face_detected": False,
        "emotions": {}
    }

@app.route("/api/detect-emotion", methods=['POST'])
def detect_emotion():
    if 'image' not in request.json:
        return jsonify({
            "success": False,
            "error": "No image data provided"
        }), 400
    
    try:
        # Decode the base64 image
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if not already
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
    return "Hello, World!"

@app.route("/api/tts", methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        speed = max(0.7, min(2.0, float(data.get('speed', 1.0))))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400

        print(f"Processing TTS request - text: {text}, voice: {voice}, speed: {speed}")

        # Get SSE client and configure it
        sse = client.tts.SSEClient()
        sse.speed = speed  # Set speed on the client
        response = sse.send(text)  # Send just the text
        
        # Create a temporary file to save the audio
        temp_file = io.BytesIO()
        
        # Save the audio stream to the temporary file
        save_audio(response, temp_file)
        
        # Debug info
        temp_file.seek(0)
        audio_data = temp_file.read()
        print(f"Generated audio size: {len(audio_data)} bytes")
        print(f"First 4 bytes: {audio_data[:4]}")
        
        # Reset file pointer for sending
        temp_file.seek(0)
        
        # Send the audio file directly with explicit headers
        response = send_file(
            temp_file,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='speech.wav'
        )
        
        # Add CORS headers explicitly
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response
    
    except Exception as e:
        print("TTS Error:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/test", methods=['GET'])
def test_endpoint():
    return jsonify({"status": "ok", "message": "Emotion detection API is running"})

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, host='0.0.0.0', port=port)