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
from database import init_db, insert_snapshot, get_all_snapshots, get_available_cameras

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files after ensuring the directory exists
static_dir = Path("app/static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

init_db()
# Initialize webcams: primary (0) and secondary (USB, 1)
camera0 = cv2.VideoCapture(0)
camera1 = cv2.VideoCapture(1)
camera2 = cv2.VideoCapture(2)

# Create snapshot directory if it doesn't exist
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Object detection count tracker
object_counts = Counter()

# Store snapshots in memory (in production, use a database)
snapshots = []

def gen_frames(camera, cam_id: str | int = 0):
    """Yield MJPEG frames from a given camera. Saves snapshots into per-camera subfolders.

    Args:
        camera: OpenCV VideoCapture instance
        cam_id: identifier used for snapshot subfolder and saved paths
    """
    last_snapshot_time = datetime.now()
    snapshot_interval = 5  # seconds

    # Ensure per-camera snapshot folder exists
    cam_str = str(cam_id)
    cam_snapshot_dir = SNAPSHOT_DIR / cam_str
    cam_snapshot_dir.mkdir(exist_ok=True)

    while True:
        success, frame = camera.read()
        if not success:
            continue

        annotated_frame, results = detect_from_frame(frame)

        # Take snapshot every snapshot_interval seconds
        current_time = datetime.now()
        time_since_last_snapshot = (current_time - last_snapshot_time).total_seconds()

        if time_since_last_snapshot >= snapshot_interval:
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = cam_snapshot_dir / f"snapshot_{timestamp_str}.jpg"
            cv2.imwrite(str(snapshot_path), annotated_frame)

            detections = {}
            if results and getattr(results, 'boxes', None):
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    detections[class_name] = True
                    object_counts[class_name] += 1

            # Store in DB using a path that includes camera id
            insert_snapshot(
                current_time.isoformat(),
                f"/snapshots/{cam_str}/snapshot_{timestamp_str}.jpg",
                detections,
                cam_str
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
    """Legacy/default video stream (primary camera)."""
    return StreamingResponse(
        gen_frames(camera0, 0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/video0")
def video0():
    """Stream from camera index 0."""
    return StreamingResponse(
        gen_frames(camera0, 0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/video1")
def video1():
    """Stream from camera index 1 (USB webcam)."""
    return StreamingResponse(
        gen_frames(camera1, 1),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/video2")
def video2():
    """Stream from camera index 2 (USB webcam)."""
    return StreamingResponse(
        gen_frames(camera2, 2),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/counts")
async def get_counts():
    return JSONResponse(object_counts)

@app.get("/get_snapshots")
async def get_snapshots(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    cam: Optional[str] = Query(None, description="Camera filter (e.g., '0' or '1')")
):
    return JSONResponse(get_all_snapshots(start_date, end_date, cam))

@app.get("/snapshots/{cam}/{filename}")
async def serve_snapshot(cam: str, filename: str):
    # Serve snapshots from per-camera folders: /snapshots/<cam>/<filename>
    return FileResponse(SNAPSHOT_DIR / cam / filename)


@app.get('/available_cameras')
async def available_cameras():
    """Return a JSON list of available camera ids detected in the DB (e.g. ['0','1'])."""
    cams = get_available_cameras()
    return JSONResponse(cams)
