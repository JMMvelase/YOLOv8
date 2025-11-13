from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from collections import Counter
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional

import cv2
import os
import json

from app.detector import detect_from_frame  # Ensure this is implemented properly in your project
from database import init_db, insert_snapshot, get_all_snapshots

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files after ensuring the directory exists
static_dir = Path("app/static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

init_db()
# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Warning: Could not open video stream.")
    camera = None

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
        if camera:
            success, frame = camera.read()
            if not success:
                continue
        else:
            # Create a black frame if no camera
            frame = cv2.UMat(480, 640, cv2.CV_8UC3)
            frame.setTo([0, 0, 0])

        annotated_frame, results = detect_from_frame(frame)

        # Take snapshot every 5 seconds
        current_time = datetime.now()
        if (current_time - last_snapshot_time).total_seconds() >= 5:
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = SNAPSHOT_DIR / f"snapshot_{timestamp_str}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)

            detections = {}
            if results and results.boxes:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    detections[class_name] = True
                    object_counts[class_name] += 1

            # Store in DB
            insert_snapshot(
                current_time.isoformat(),
                f"/snapshots/snapshot_{timestamp_str}.jpg",
                detections
            )

            last_snapshot_time = current_time

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
    return templates.TemplateResponse("history.html", {"request": request})


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
async def get_snapshots(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    return JSONResponse(get_all_snapshots(start_date, end_date))

@app.get("/snapshots/{filename}")
async def serve_snapshot(filename: str):
    return FileResponse(SNAPSHOT_DIR / filename)
