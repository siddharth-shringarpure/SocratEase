from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pyneuphonic import Neuphonic, save_audio
import io
from dotenv import load_dotenv
import traceback

# Debug information about Python environment
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("PYTHONPATH:", sys.path)
print("Current working directory:", os.getcwd())
print("Environment variables:", dict(os.environ))

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5328))
    app.run(debug=True, port=port)