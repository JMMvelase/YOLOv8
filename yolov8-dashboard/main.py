from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from collections import Counter
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import cv2
import os
import json

from app.detector import detect_from_frame  # Ensure this is implemented properly in your project


app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize webcam
camera = cv2.VideoCapture(0)

# Create snapshot directory if it doesn't exist
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Object detection count tracker
object_counts = Counter()

# Store snapshots in memory (in production, use a database)
snapshots = []

def gen_frames():
    last_snapshot_time = datetime.now()
    
    while True:
        success, frame = camera.read()
        if not success:
            continue

        # Run YOLO detection
        annotated_frame, results = detect_from_frame(frame)

        # Take snapshot every 5 seconds
        current_time = datetime.now()
        if (current_time - last_snapshot_time).total_seconds() >= 5:
            # Save snapshot
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)
            
            # Get detections from YOLO results
            detections = {}
            if results and results.boxes:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    detections[class_name] = True
                    object_counts[class_name] += 1
            
            # Add snapshot to memory
            snapshots.append({
                "timestamp": current_time.isoformat(),
                "image_url": f"/snapshots/snapshot_{timestamp}.jpg",
                "detections": detections
            })
            
            # Keep only last 100 snapshots
            if len(snapshots) > 100:
                old_snapshot = snapshots.pop(0)
                old_path = SNAPSHOT_DIR / Path(old_snapshot["image_url"]).name
                if old_path.exists():
                    old_path.unlink()
            
            last_snapshot_time = current_time

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/employee.html", response_class=HTMLResponse)
async def employee(request: Request):
    return templates.TemplateResponse("employee.html", {"request": request})
@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("/dashboard.html", {"request": request})
@app.get("/history.html", response_class=HTMLResponse)
async def history(request: Request):    
    return templates.TemplateResponse("/history.html", {"request": request})


@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/counts")
async def get_counts():
    return JSONResponse(object_counts)

@app.get("/get_snapshots")
async def get_snapshots():
    return JSONResponse(snapshots)

@app.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    return FileResponse(SNAPSHOT_DIR / filename)
