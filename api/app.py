"""FastAPI application entry point for the speech analysis API."""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.feedback_router import feedback_router
from api.routers.file_router import file_router
from api.routers.misc_router import misc_router
from api.routers.transcription_router import transcription_router
from api.routers.vision_router import vision_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down application resources."""
    Path("temp").mkdir(exist_ok=True)
    Path("uploads").mkdir(exist_ok=True)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Initialising Whisper model on device: %s", device)
        whisper.load_model("small").to(device)
        logging.info("✓ Whisper model loaded on %s", device)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Failed to load Whisper model: %s", e, exc_info=True)

    yield


app = FastAPI(title="SocratEase API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(misc_router)
app.include_router(feedback_router)
app.include_router(file_router)
app.include_router(transcription_router)
app.include_router(vision_router)
