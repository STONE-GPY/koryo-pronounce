import os
import shutil
import uuid
import logging
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app import PronunciationApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Koryo-saram Pronunciation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
pronunciation_app = PronunciationApp()
os.makedirs("data/uploads", exist_ok=True)

@app.post("/api/analyze_whisperx", response_model=Dict[str, Any])
async def analyze_whisperx(audio: UploadFile = File(...), target_text: str = Form(...)) -> Dict[str, Any]:
    """Analyzes the pronunciation using WhisperX."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    file_id = str(uuid.uuid4())
    ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
    file_path = f"data/uploads/{file_id}.{ext}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Run WhisperX analysis
        result = pronunciation_app.analyze_with_whisperx(file_path, target_text)
        return result
    except Exception as e:
        logger.error(f"Error during WhisperX analysis: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error during analysis."})
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")

@app.post("/api/analyze", response_model=Dict[str, Any])
async def analyze(audio: UploadFile = File(...), target_text: str = Form(...)) -> Dict[str, Any]:
    """Analyzes the pronunciation of the uploaded audio against the target text."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")
        
    file_id = str(uuid.uuid4())
    ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
    file_path = f"data/uploads/{file_id}.{ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Run analysis pipeline
        result = pronunciation_app.analyze_pronunciation(file_path, target_text)
        return result
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error during analysis."})
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")

# Serve the static UI files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
