import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app import PronunciationApp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
pronunciation_app = PronunciationApp()
os.makedirs("data/uploads", exist_ok=True)

@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...), target_text: str = Form(...)):
    file_id = str(uuid.uuid4())
    ext = audio.filename.split(".")[-1] if "." in audio.filename else "webm"
    file_path = f"data/uploads/{file_id}.{ext}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    try:
        # Run analysis pipeline
        result = pronunciation_app.analyze_pronunciation(file_path, target_text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

# Serve the static UI files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
