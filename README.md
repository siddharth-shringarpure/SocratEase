# SocratEase

**Speak smarter with AI-powered feedback.**

SocratEase is a platform that turns speech videos into personalised, actionable feedback using advanced AI models. Whether you're preparing for a job interview, refining your public speaking, or practising everyday conversations, SocratEase analyses your speech, facial expressions, and engagement levels to offer valuable, personalised feedback.

Using digital image processing, natural language processing (NLP), and audio signal processing, SocratEase evaluates communication across three dimensions: visual cues (facial expressions and eye contact), auditory features (speech fluency), and textual analysis (logical coherence and engagement). These are combined through a late-fusion multimodal approach, as described in [D'Mello 2015](https://dl.acm.org/doi/pdf/10.1145/2682899), to produce a comprehensive evaluation of your speaking style.

## Features

**Video Feed Analysis** — Evaluate engagement through eye contact and facial expressions.

- **Eye Contact Detection** — Tracks gaze direction using MediaPipe face landmarks
- **Facial Expression Analysis** — Identifies emotions and microexpressions via DeepFace

**Audio Feed Analysis** — Focuses on fluency features in speech.

- **Fluency Metrics** — Implements techniques from [Eusipco 2023](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0000231.pdf)
- **Dataset** — Uses the [Avalinguo-Audio-Set](https://github.com/agrija9/Avalinguo-Audio-Set)
- **Speech Features Extracted**:
  - Words per Minute
  - Lexical Density (Token Type Ratio)
  - Zero Crossing Rate (silent pauses)
  - MFCC (Mel-frequency cepstral coefficients)
  - Features are trained on an XGBoost model ([Chen 2016](https://dl.acm.org/doi/10.1145/2939672.2939785)) achieving 93% F1-score

**Communication Transcript Analysis** — Examines speech patterns, coherence, and engagement.

- **Tonality Analysis** — Uses a [tone analysis dataset](https://www.kaggle.com/datasets/sameedatif/tone-analysis) + [3gpp-embedding-model-v0](https://huggingface.co/iris49/3gpp-embedding-model-v0) + MLP
- **Filler Word Frequency** — Computes corpus occurrence in the transcript
- **Vocabulary Sophistication** — Assesses Type Token Ratio (TTR)
- **Logical Flow Detection** — Uses [roberta-large_overall-coherence](https://huggingface.co/SushantGautam/roberta-large_overall-coherence)
- **Engagement Prediction** — Flesch-Kincaid Readability score

## Tech Stack

- **Frontend**: Next.js, React, TailwindCSS, Shadcn UI
- **Backend**: FastAPI (Python), Next.js API routes
- **AI/ML**:
  - Computer Vision: MediaPipe, OpenCV, DeepFace
  - NLP: NLTK, HuggingFace Transformers, PyTorch
  - Audio: Librosa, Pydub, supertonic-3 (local TTS), Whisper
- **Deployment**: Docker, Docker Compose

## What's Next

- **Multilingual Support** — Expanding accessibility for diverse linguistic backgrounds
- **Enhanced Emotion Detection** — Refining sentiment analysis and engagement prediction
- **Real-Time Fluency Feedback** — Instant analysis with actionable recommendations
- **Context-Aware Coherence** — Improving logical flow detection with stronger reasoning models
- **Communication Analytics** — Detailed performance tracking over time

## Development Setup

### Prerequisites

- Node.js 18+ and pnpm
- Python 3.10 (specifically 3.10.x for MediaPipe compatibility)
  - Fedora/RHEL: `sudo dnf install python3.10-devel`
  - Ubuntu/Debian: `sudo apt install python3.10-dev`
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker and Docker Compose (optional)
- FFmpeg
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- PortAudio
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
   pnpm install
   uv sync
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Download required models:

   ```bash
   pnpm setup
   ```

### Running the Application

#### Option 1: Local development (recommended)

```bash
# Terminal 1: FastAPI backend
pnpm backend-dev

# Terminal 2: Next.js frontend
pnpm dev
```

Frontend: http://localhost:3000  
Backend API: http://localhost:5000

#### Option 2: Docker Compose

```bash
pnpm docker:dev
```

```bash
# Individual Docker commands
docker compose build --no-cache
docker compose up
docker compose up -d     # background
docker compose logs -f
docker compose down
```

### Project Structure

```
socratease/
├── api/                  # FastAPI backend
│   ├── routers/          # API route handlers
│   ├── services/         # ML logic and processing
│   ├── core/             # Config, constants, shared models
│   ├── utils/            # Utility functions
│   └── app.py            # Application entry point
├── app/                  # Next.js frontend
│   ├── api/              # Next.js API routes
│   └── components/       # React components
├── public/               # Static assets and frontend models
├── scripts/              # Dev and utility scripts
└── uploads/              # User recordings (gitignored)
```

### Architecture

1. **Frontend (Next.js)** — UI rendering, client-side logic, proxies requests to the FastAPI backend via Next.js rewrites
2. **FastAPI Backend** — Core ML/AI processing: computer vision, audio analysis, NLP, TTS feedback
3. **Data Flow** — Recordings are uploaded to the backend, processed server-side, results returned to the frontend

### Troubleshooting

**MediaPipe install fails**

```bash
pip install mediapipe==0.10.21
```

**Port conflicts** — edit `docker-compose.yml` to change mapped ports, or check what's using 3000/5000.

**Docker containers fail to start**

```bash
docker compose logs backend
docker compose down && docker compose build --no-cache
```
