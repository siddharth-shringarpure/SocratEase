# SocratEase

## SocratEase: Speak smarter with AI-powered feedback.

SocratEase is a magical platform designed to turn your speech videos into insightful feedback through advanced AI models. Whether you're preparing for a job interview, perfecting your public speaking skills, or mastering your first date conversations, SocratEase analyses your speech, facial expressions, and engagement levels to offer valuable, personalised feedback.

Using a combination of digital image processing, natural language processing (NLP), and audio signal processing, SocratEase evaluates your communication on three key dimensions: visual cues (like facial expressions and eye contact), auditory features (such as speech fluency), and textual analysis (for logical coherence and engagement). By combining these analyses through a late-fusion multimodal approach, SocratEase provides an integrated understanding of your speaking style and areas for improvement.

This approach, detailed in [D'Mello 2015](https://dl.acm.org/doi/pdf/10.1145/2682899), allows us to process each modality independently before combining them to generate actionable feedback; the final result is a comprehensive evaluation of your communication skills, helping you feel more confident and prepared for any speaking situation.

## Features

🎥 **Video Feed Analysis** – Evaluate user engagement through eye contact and facial expressions

- 👀 **Eye Contact Detection** – Improve users' engagement in conversations and speeches using [GazeTracking](https://github.com/antoinelame/GazeTracking)'s method
- 😀 **Facial Expression Analysis** – Identifies emotions and microexpressions by using techniques from [Edlitera](https://www.edlitera.com/blog/posts/emotion-detection-in-video)

🎙️ **Audio Feed Analysis** – Focuses on fluency features in speech.

- 🗣 **Fluency Metrics** – Implements techniques from [Eusipco 2023](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000231.pdf).
- 📊 **Dataset Utilisation** – Uses the [Avalinguo-Audio-Set](https://github.com/agrija9/Avalinguo-Audio-Set).
- 🔎 **Speech Features Extracted**:
  - ⏱ **Words per Minute**
  - 📖 **Lexical Density** (Token Type Ratio)
  - 🔕 **Zero Crossing Rate** (silent pauses)
  - 🎵 **MFCC** (Mel-frequency cepstral coefficients)
  - The features are extracted, **trained on an extreme gradient boosting (XGB) model** [Chen 2016](https://dl.acm.org/doi/10.1145/2939672.2939785) to predict the **fluency**
    - The XGB model is trained using Randomised CV model selection, achieving 93% overall F1-Score
    - Alongside other features, fluency is also used as a feedback to the user

📝 **Communication Transcript Analysis** – Examines speech patterns, coherence, and engagement.

- ✍️ **Tonality Analysis** – Utilises [tone analysis dataset](https://www.kaggle.com/datasets/sameedatif/tone-analysis) + [
  3gpp-embedding-model-v0](https://huggingface.co/iris49/3gpp-embedding-model-v0) + MLP
- 🚫 **Filler Word Frequency** – Computes corpus occurrence in the speech's transcript.
- 📚 **Vocabulary Sophistication** – Assesses Type Token Ratio (TTR) to see how much the words are repeated
- 🔄 **Logical Flow Detection** – Leverages [roberta-large_overall-coherence](https://huggingface.co/SushantGautam/roberta-large_overall-coherence) to use logistic regression in finding the coherence of the speech
- 🎭 **Engagement Prediction** – Uses **Flesch-Kincaid Readability** to estimate listener interests

## Implementation

### Tech Stack

- **Frontend**: Next.js, React, TailwindCSS, Shadcn UI
- **Backend**: Flask (Python), Next.js API routes
- **AI/ML Processing**:
  - Computer Vision: MediaPipe, OpenCV, FER (Facial Emotion Recognition)
  - NLP: NLTK, HuggingFace Transformers, PyTorch
  - Audio Processing: Librosa, Pydub, Neuphonic
- **Deployment**: Docker, Docker Compose
- **Data Storage**: File system (for recordings and processed results)

## What's Next for SocratEase?

As we continue enhancing our analysis system, we plan to introduce new intelligent features and improvements, including:

🌍 **Multilingual Speech & Text Support** – Expanding accessibility for diverse linguistic backgrounds.

🎯 **Enhanced Emotion & Engagement Detection** – Refining sentiment analysis and listener interest prediction for more accurate insights.

🎙️ **Real-Time Fluency Feedback** – Providing instant analysis of speech fluency with actionable recommendations.

🕵️ **Context-Aware Coherence Evaluation** – Improving logical flow detection with more robust reasoning models.

📊 **Comprehensive Communication Analytics** – Introducing detailed performance tracking and insights for continuous improvement.

## Development Setup

### Prerequisites

- Node.js 18+ and npm
- Python 3.10 (specifically 3.10.x for MediaPipe compatibility)
  - **Important**: Install development headers
    - Fedora/RHEL/CentOS: `sudo dnf install python3.10-devel`
    - Ubuntu/Debian: `sudo apt install python3.10-dev`
- Docker and Docker Compose (optional, for containerised development)
- FFmpeg (for audio processing)
- PortAudio development libraries (for PyAudio)
  - macOS: `brew install portaudio`
  - Fedora/RHEL: `sudo dnf install portaudio-devel`
  - Ubuntu/Debian: `sudo apt install portaudio19-dev`

### Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/siddharth-sh/socratease.git
   cd socratease
   ```

2. Install dependencies:

   ```bash
   # Install frontend dependencies
   npm install

   # For Python backend (if running locally)
   # Ensure you're using Python 3.10
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac or .\venv\Scripts\activate on Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   ```bash
   # Root environment variables
   cp .env.example .env  # Main backend config and API keys

   # Frontend environment variables
   cp frontend/.env.local.example frontend/.env.local  # Frontend-specific API URL
   ```

4. Download required models:

   ```bash
   # Download all required models (frontend and backend)
   npm run setup

   # Or individually:
   npm run download-models        # Frontend models
   npm run download-backend-models # Backend ML models
   ```

### Running the Application

#### Option 1: Using npm Scripts (Recommended)

```bash
# Run Next.js with Docker backend
npm run docker:dev

# Fast development mode (Flask backend runs locally)
npm run dev:fast

# Just the Next.js frontend (if backend is already running)
npm run dev
```

#### Option 2: Docker Compose Commands (for backend)

```bash
# Build the backend
docker compose build --no-cache

# Run the backend
docker compose up

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

#### Option 3: Manual Setup for Development

```bash
# Terminal 1: Start the Flask backend
source venv/bin/activate  # On Linux/Mac or .\venv\Scripts\activate on Windows
cd api
python app.py

# Terminal 2: Start the Next.js frontend
npm run next-dev
```

The application will be available at:

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

### Architecture Overview

SocratEase uses a hybrid architecture:

1. **Frontend (Next.js)**:

   - Handles UI rendering and client-side logic
   - Communicates with both Next.js API routes and Flask backend

2. **Next.js API Routes**:

   - Act as a proxy for health checks
   - Handle simple API functionality

3. **Flask Backend**:

   - Processes the core ML/AI tasks
   - Manages complex processing operations like:
     - Computer vision (gaze tracking, emotion detection)
     - Audio analysis
     - Natural language processing

4. **Data Flow**:
   - User recordings are sent to the Flask backend
   - Processing happens on the server side
   - Results are returned to frontend

### Project Structure

```
socratease/
├── api/              # Python Flask backend
│   ├── routes/       # API endpoints for different features
│   ├── services/     # Core ML logic and processing
│   └── app.py        # Main Flask server
├── app/              # Next.js frontend
│   ├── api/          # Next.js API routes
│   ├── components/   # React components
│   └── pages/        # Application pages
├── public/           # Static assets
├── scripts/          # Development and utility scripts
└── uploads/          # User uploaded files (gitignored)
```

### Troubleshooting

#### Python and Dependencies

If you encounter errors installing dependencies:

1. Ensure you're using Python 3.10:

   ```bash
   python3.10 --version  # Should show Python 3.10.x
   ```

2. Try installing MediaPipe separately:

   ```bash
   pip install mediapipe==0.10.0
   ```

3. Make sure OpenCV is properly installed:

   ```bash
   pip install opencv-python==4.8.0.74
   ```

4. For issues with audio processing:

   ```bash
   # Ensure FFmpeg is installed on your system
   # Ubuntu/Debian:
   sudo apt install ffmpeg

   # macOS:
   brew install ffmpeg

   # Windows: Download from https://ffmpeg.org/download.html
   ```

5. For `portaudio.h file not found` error:

   ```bash
   # macOS:
   brew install portaudio

   # Fedora/RHEL:
   sudo dnf install portaudio-devel

   # Ubuntu/Debian:
   sudo apt install portaudio19-dev
   ```

#### Docker Issues

1. If containers fail to start:

   ```bash
   # Check container logs
   docker compose logs backend

   # Try rebuilding containers
   docker compose down
   docker compose build --no-cache
   ```

2. For port conflicts:
   - Edit the docker-compose.yml file to change mapped ports
   - Check for processes using ports 3000 or 5000
