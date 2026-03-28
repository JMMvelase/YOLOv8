from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from
load_dotenv()

from app.detector import detect_from_upload

app = FastAPI()
templates_dir = Path(__file__).parent / "app" / "templates"

# Mount static files
static_dir = Path(__file__).parent / "app" / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir.as_posix()), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with webcam detection"""
    html_file = templates_dir / "dashboard.html"
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard():
    """Dashboard page"""
    html_file = templates_dir / "dashboard.html"
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))

@app.get("/employee.html", response_class=HTMLResponse)
async def employee():
    """Employee page"""
    html_file = templates_dir / "employee.html"
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))

@app.get("/history.html", response_class=HTMLResponse)
async def history():
    """History page"""
    html_file = templates_dir / "history.html"
    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))

@app.post("/detect-webcam")
async def detect_webcam_frame(image: UploadFile = File(...)):
    """Detect PPE from webcam frame"""
    try:
        contents = await image.read()
        result = detect_from_upload(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "predictions": []},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    weights_path = Path(__file__).parent / "weights" / "best.1.0.pt"
    model_loaded = weights_path.exists()
    
    return {
        "status": "ok",
        "model_type": "YOLOv8 (Local)",
        "weights_file": "best.1.0.pt",
        "model_loaded": model_loaded
    }



