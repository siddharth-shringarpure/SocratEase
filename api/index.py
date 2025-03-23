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
import re
from itertools import tee
# import t
import pickle as pkl
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

# Add after the imports
FILLER_WORDS = [
    "um", "uh", "like", "you know", "well", "so", "actually", "basically", "i mean", 
    "right", "okay", "er", "hmm", "literally", "anyway", "of course", "i guess", 
    "in other words", "obviously", "to be honest", "just", "seriously", "you see", 
    "i suppose", "frankly", "well, i mean", "at the end of the day", "to tell the truth", 
    "as it were", "kind of", "sort of", "in a way", "that is", "as a matter of fact", 
    "in fact", "like i said", "more or less", "i don't know", "basically speaking", 
    "for sure", "you could say", "the thing is", "it s like", "put it another way", 
    "at least", "as such", "well you know", "i would say", "truth be told", "yeah", "and yeah",
    "um yeah", "um no", "um right", "like literally", "to", "erm", "let s see", "hm", "maybe",
    "maybe like"
]

def ngrams(words, n):
    output = []
    for i in range(len(words) - n + 1):
        output.append(' '.join(words[i:i + n]))
    return output

def analyse_filler_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    # Count filler words
    filler_count = 0
    found_fillers: list[str] = []
    
    for word in words:
        if word in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(word)
    
    bigrams = ngrams(words, 2)
    for bigram in bigrams:
        if bigram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(bigram)
    
    trigrams = ngrams(words, 3)
    for trigram in trigrams:
        if trigram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(trigram)
    
    quadgrams = ngrams(words, 4)
    for quadgram in quadgrams:
        if quadgram in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(quadgram)
    
    # Calculate metrics
    filler_percentage = (filler_count / total_words * 100) if total_words > 0 else 0
    
    # Determine emoji based on percentage
    if filler_percentage <= 3:
        emoji = "🎯"  # Excellent
    elif filler_percentage <= 7:
        emoji = "👍"  # Good
    elif filler_percentage <= 12:
        emoji = "💭"  # Think about it
    elif filler_percentage <= 18:
        emoji = "⚠️"  # Warning
    else:
        emoji = "😞"  # Needs work
    
    # Calculate TTR
    ttr_analysis = calculate_ttr(text)
    
    # Get logical flow score
    try:
        logical_score = logical_flow(text)
        # Convert logical score to percentage and determine emoji
        logical_percentage = logical_score * 100
        if logical_percentage >= 80:
            logical_emoji = "🌠"  # Excellent flow
        elif logical_percentage >= 60:
            logical_emoji = "🌊"  # Good flow
        elif logical_percentage >= 40:
            logical_emoji = "🔄"  # Average flow
        elif logical_percentage >= 20:
            logical_emoji = "🌫️"  # Needs improvement
        else:
            logical_emoji = "🌪️"  # Poor flow
    except Exception as e:
        print(f"Error calculating logical flow: {str(e)}")
        logical_percentage = 0
        logical_emoji = "❓"
    
    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_percentage": round(filler_percentage, 2),
        "found_fillers": found_fillers,
        "filler_emoji": emoji,
        "ttr_analysis": ttr_analysis,
        "logical_flow": {
            "score": round(logical_percentage, 2),
            "emoji": logical_emoji
        }
    }

def calculate_ttr(text):
    words = re.findall(r'\b\w+\b', text.lower())
    total_words: int = len(words)
    unique_words: int = len(set(words))
    ttr: float = (unique_words / total_words) * 100 if total_words > 0 else 0
    
    # Determine diversity level and emoji
    if ttr >= 80:
        diversity = "very high"
        emoji = "🌟"
    elif ttr >= 60:
        diversity = "high"
        emoji = "✨"
    elif ttr >= 40:
        diversity = "average"
        emoji = "💫"
    elif ttr >= 20:
        diversity = "low"
        emoji = "📝"
    else:
        diversity = "very low"
        emoji = "📚"
        
    return {
        "ttr": round(ttr, 2),
        "unique_words": unique_words,
        "diversity_level": diversity,
        "emoji": emoji
    }

def logical_flow(text):
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'logical_model.pk')
        print(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            return 0.0
        
        # Check PyTorch version
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check transformers version
        import transformers
        print(f"Transformers version: {transformers.__version__}")
            
        with open(model_path, 'rb') as f:
            try:
                logical_model = pkl.load(f)
                print("Successfully loaded logical flow model")
            except RuntimeError as e:
                if "register_pytree_node()" in str(e):
                    print("Version mismatch detected between PyTorch and transformers")
                    print("Please ensure compatible versions are installed")
                    return 0.0
                raise
        
        print(f"Making prediction for text of length: {len(text)}")
        pred: list[dict] = logical_model.predict(text)
        score: float = pred[0]['score']
        print(f"Logical flow prediction result: {score}")
        
        return score
    except Exception as e:
        print(f"Error in logical flow prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
        return 0.0  # Return 0 as fallback to indicate failure

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

def transcribe_long_audio(audio_path, max_duration=30):
    """
    Transcribe longer audio files by splitting into chunks if needed.
    Returns the full transcription text.
    """
    try:
        # Get audio duration using ffprobe
        duration_cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration = float(duration_result.stdout.strip())
        print(f"Audio duration: {duration} seconds")
        
        # If short enough, transcribe directly
        if duration <= max_duration:
            result = model.transcribe(audio_path)
            return result.get("text", "")
            
        # For longer files, process in chunks
        chunks = []
        for start in range(0, int(duration), max_duration):
            chunk_path = f"{audio_path}_chunk_{start}.mp3"
            
            # Extract chunk with 1-second overlap
            chunk_cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-ss', str(max(0, start - 1)),  # Start 1 second earlier for overlap
                '-t', str(max_duration + 1),     # Add 1 second for overlap
                '-acodec', 'libmp3lame',
                '-ar', '16000',
                '-ac', '1',
                chunk_path
            ]
            subprocess.run(chunk_cmd, capture_output=True)
            
            try:
                # Transcribe chunk with a simpler timeout mechanism
                import threading
                import queue
                
                def transcribe_chunk():
                    result = model.transcribe(chunk_path)
                    return result.get("text", "")
                
                # Create a queue to store the result
                result_queue = queue.Queue()
                
                # Start transcription in a separate thread
                thread = threading.Thread(target=lambda: result_queue.put(transcribe_chunk()))
                thread.start()
                
                # Wait for result with timeout
                try:
                    text = result_queue.get(timeout=30)  # 30 second timeout
                    chunks.append(text)
                except queue.Empty:
                    print(f"Timeout transcribing chunk starting at {start} seconds")
                    continue
                
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    
            except Exception as e:
                print(f"Error processing chunk {start}: {str(e)}")
                continue
        
        # Combine chunks with overlap handling
        if not chunks:
            return ""
            
        # Join chunks with overlap handling
        combined_text = " ".join(chunks)
        # Remove duplicate words at chunk boundaries
        words = combined_text.split()
        cleaned_words = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned_words.append(word)
        
        return " ".join(cleaned_words)
        
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

@app.route("/api/speech2text", methods=['POST'])
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
        # Save and verify input file
        file.save(temp_input)
        input_size = os.path.getsize(temp_input)
        print(f"Saved input file to {temp_input} (size: {input_size} bytes)")
        
        if input_size == 0:
            raise Exception("Input file is empty")

        try:
            # Extract audio
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-i', temp_input,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ar', '16000',
                '-ac', '1',
                '-b:a', '192k',
                '-af', 'highpass=f=50,lowpass=f=15000,volume=2,afftdn=nf=-20',
                temp_audio
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            if not os.path.exists(temp_audio):
                raise Exception("Failed to create audio file")
            
            # Use the new transcription function
            text = transcribe_long_audio(temp_audio)
            if not text:
                raise Exception("Failed to transcribe audio")
            
            # Analyze the text
            analysis = analyse_filler_words(text)
            
            return jsonify({
                "text": text,
                "analysis": analysis
            })
            
        except Exception as e:
            print(f"Processing Error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temp files
            for temp_file in [temp_input, temp_audio]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Failed to clean up {temp_file}: {str(e)}")
                    
    except Exception as e:
        print(f"General Error: {str(e)}")
        traceback.print_exc()
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