import logging
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional

from agent.voice_agent import VoiceAgent

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Receptionist API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Voice Agent
# Note: This loads models into memory on startup
agent = VoiceAgent()

# Directories for audio files
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the output_audio directory to serve generated WAV files
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

class ProcessResponse(BaseModel):
    transcription: str
    language: str
    intent: str
    entities: Dict[str, Any]
    response_text: str
    audio_url: str
    status: str

@app.post("/process-voice", response_model=ProcessResponse)
async def process_voice(file: UploadFile = File(...)):
    """
    Receives an audio file, runs it through the voice agent pipeline,
    and returns the structured results + audio response URL.
    """
    session_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing voice request: {input_path}")
        
        # Run through the standardized pipeline
        result = agent.process_audio(input_path)
        
        # Map to ProcessResponse
        audio_url = f"/audio/{os.path.basename(result['audio_path'])}" if result['audio_path'] else ""
        
        return ProcessResponse(
            transcription=result["transcription"],
            language=result["language"],
            intent=result["intent"],
            entities=result["entities"],
            response_text=result["response_text"],
            audio_url=audio_url,
            status="success" if result["transcription"] else "no_speech"
        )

    except Exception as e:
        logger.error(f"Error processing voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
