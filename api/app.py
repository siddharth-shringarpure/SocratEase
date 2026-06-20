"""
Flask application entry point for the speech analysis API.

This module initialises the Flask application, configures logging and CORS,
loads environment variables, and sets up the Whisper model for speech recognition.
"""
import logging
import os
import sys

import torch
import whisper
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

# Load environment configuration
load_dotenv()

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialise Whisper model before creating Flask app
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Initialising Whisper model on device: %s", device)
    model = whisper.load_model("small").to(device)
    logging.info("Whisper model loaded successfully on %s", device)
except Exception as e:
    logging.error("Failed to load Whisper model: %s", e, exc_info=True)
    model = None

# Initialise Flask app
app = Flask(__name__)

# Set up required storage directories
os.makedirs("temp", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Configure cross-origin resource sharing
CORS(app, resources={r"/*": {
    "origins": ["*"],  # TODO: Restrict to specific domains in production
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["*"],
    "supports_credentials": True,
    "expose_headers": ["*"]
}})

# Register route blueprints
from api.routes.feedback_routes import feedback_bp
from api.routes.file_routes import file_bp
from api.routes.misc_routes import misc_bp
from api.routes.transcription_routes import transcription_bp
from api.routes.vision_routes import vision_bp

app.register_blueprint(misc_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(file_bp)
app.register_blueprint(transcription_bp)
app.register_blueprint(vision_bp)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
