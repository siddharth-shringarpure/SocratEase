from flask import Blueprint, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from PIL import Image
import io
import base64
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig

app = Blueprint('emotion_detection', __name__)
CORS(app)

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize MTCNN model for face detection
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=100,  # Lowered to detect faces more easily
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    keep_all=False,
    device=device
)

# Load the pre-trained model and feature extractor
MODEL_NAME = "trpakov/vit-face-expression"
print("Loading ViT-Face-Expression model...")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
id2label = config.id2label
print("Model loaded successfully!")

def detect_emotions(image):
    """
    Detect emotions from a given image.
    Returns a tuple of the cropped face image and a dictionary of class probabilities.
    """
    # Detect faces in the image using MTCNN
    sample = mtcnn.detect(image)
    
    if sample[0] is not None and len(sample[0]) > 0:
        box = sample[0][0]
        
        # Convert box coordinates to integers
        box = [int(coord) for coord in box]
        
        # Crop the face
        face = image.crop(box)
        
        # Convert face to base64 for response
        face_bytes = io.BytesIO()
        face.save(face_bytes, format='JPEG')
        face_base64 = base64.b64encode(face_bytes.getvalue()).decode('utf-8')
        
        # Pre-process the face
        inputs = extractor(images=face, return_tensors="pt")
        
        # Run the image through the model
        outputs = model(**inputs)
        
        # Apply softmax to the logits to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert probabilities tensor to a Python list
        probabilities = probabilities.detach().numpy().tolist()[0]
        
        # Map class labels to their probabilities
        class_probabilities = {
            id2label[i]: float(prob) for i, prob in enumerate(probabilities)
        }
        
        return {
            "success": True,
            "face_detected": True,
            "face_image": face_base64,
            "emotions": class_probabilities
        }
    
    return {
        "success": True,
        "face_detected": False,
        "emotions": {}
    }

@app.route('/detect-emotion', methods=['POST'])
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
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500 