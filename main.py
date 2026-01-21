from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from
load_dotenv()

from app.detector import detect_from_upload

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files
static_dir = Path("app/static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with webcam detection"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/employee.html", response_class=HTMLResponse)
async def employee(request: Request):
    """Employee page"""
    return templates.TemplateResponse("employee.html", {"request": request})

@app.get("/history.html", response_class=HTMLResponse)
async def history(request: Request):
    """History page"""
    return templates.TemplateResponse("history.html", {"request": request})

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
    api_key_set = bool(os.getenv("ROBOFLOW_API_KEY"))
    model_id = os.getenv("MODEL_ID", "ppe-detection/1")
    
    return {
        "status": "ok",
        "api_key_configured": api_key_set,
        "model_id": model_id
    }



