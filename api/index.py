from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pyneuphonic import Neuphonic, save_audio
import io
from dotenv import load_dotenv
import traceback
import whisper  # Add import for Whisper
import numpy as np
import base64
from PIL import Image
from deepface import DeepFace
import subprocess  # Add at the top with other imports

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

# Initialize Whisper model
model = whisper.load_model("base")  # You can choose "tiny", "base", "small", "medium", or "large"

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

@app.route("/api/upload-video", methods=['POST'])
def upload_video():
    """Handles video file uploads and saves them locally"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp}.mp4"
        audio_filename = f"recording_{timestamp}_audio.wav"
        video_path = os.path.join(upload_dir, video_filename)
        audio_path = os.path.join(upload_dir, audio_filename)
        
        # Save the video file
        video_file.save(video_path)
        
        # Check if video has audio stream using ffprobe
        probe_result = subprocess.run([
            'ffprobe', 
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=nw=1:nk=1',
            video_path
        ], capture_output=True, text=True)
        
        has_audio = probe_result.stdout.strip() == 'audio'
        
        if has_audio:
            try:
                # Extract audio with quality improvements
                subprocess.run([
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ac', '2',  # Stereo
                    audio_path
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print("FFmpeg error:", e.stderr)
                has_audio = False
            except FileNotFoundError:
                print("FFmpeg not found")
                has_audio = False
        
        return jsonify({
            "success": True,
            "message": "Video uploaded successfully",
            "filename": video_filename,
            "audio_filename": audio_filename if has_audio else None,
            "has_audio": has_audio
        })
        
    except Exception as e:
        print("Video Upload Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/speech2text', methods=['POST'])
def transcribe():
    """Converts speech audio to text using Whisper model"""
    try:
        print("Starting transcription request...")
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        temp_input = os.path.join(temp_dir, "temp_input.mp4")
        temp_audio = os.path.join(temp_dir, "temp_audio.mp3")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        print(f"Received file: {file.filename}, mimetype: {file.content_type}")
        file.save(temp_input)
        print(f"Saved input file to {temp_input}")

        try:
            # First, check if input file exists and has content
            if not os.path.exists(temp_input) or os.path.getsize(temp_input) == 0:
                raise Exception("Input file is empty or not created")

            print("Extracting audio from video...")
            # Extract audio with quality improvements
            result = subprocess.run([
                'ffmpeg',
                '-y',
                '-i', temp_input,
                '-vn',  # No video
                '-acodec', 'libmp3lame',  # MP3 codec
                '-ar', '44100',  # 16kHz sampling rate
                '-ac', '1',      # Mono
                '-b:a', '192k',  # Bitrate
                '-af', 'highpass=f=50,lowpass=f=15000,volume=2,afftdn=nf=-20',  # Audio filters
                temp_audio
            ], check=True, capture_output=True, text=True)
            
            print("Audio extraction complete")
            print("FFmpeg stdout:", result.stdout)
            print("FFmpeg stderr:", result.stderr)

            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                raise Exception("FFmpeg failed to extract audio")

            print("Starting Whisper transcription...")
            try:
                transcription = model.transcribe(
                    temp_audio,
                    language="en",
                    initial_prompt="This is a recording of someone speaking clearly.",
                    condition_on_previous_text=False,
                )
                
                if not transcription or not transcription.get("text"):
                    raise Exception("No transcription generated")
                
                return jsonify({"text": transcription.get("text", "")})
            except Exception as e:
                print("Whisper Error:", str(e))
                print("Traceback:", traceback.format_exc())
                return jsonify({"error": "Failed to process audio with Whisper. Please try again."}), 500

        except subprocess.CalledProcessError as e:
            print("FFmpeg Error:", e.stderr)
            return jsonify({"error": f"Failed to process audio file: {e.stderr}"}), 500
        except Exception as e:
            print("Processing Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
        finally:
            # Clean up temp files
            for temp_file in [temp_input, temp_audio]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"Cleaned up {temp_file}")
                except Exception as e:
                    print(f"Failed to clean up {temp_file}: {str(e)}")

    except Exception as e:
        print("General Error:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": "Failed to process request"}), 500

@app.route("/api/test", methods=['GET'])
def test_endpoint():
    """Health check endpoint for emotion detection API"""
    return jsonify({"status": "ok", "message": "Emotion detection API is running"})

@app.route("/uploads/<path:filename>")
def serve_file(filename):
    """Serves files from the uploads directory"""
    try:
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
            
        # Determine mime type based on extension
        if filename.endswith('.mp4'):
            mime_type = 'video/mp4'
        elif filename.endswith('.wav'):
            mime_type = 'audio/wav'
        else:
            mime_type = 'audio/mpeg'
        
        print(f"Serving file: {file_path} with mime type: {mime_type}")
        return send_file(
            file_path,
            mimetype=mime_type,
            as_attachment=False
        )
    except Exception as e:
        print("Error serving file:", str(e))
        print("Traceback:", traceback.format_exc())
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, host='0.0.0.0', port=port)