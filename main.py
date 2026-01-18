from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
from collections import Counter
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import json
import os

from app.detector import detect_from_frame
from database import init_db, insert_snapshot, get_all_snapshots, get_available_cameras

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Mount static files
static_dir = Path("app/static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

init_db()

# Camera sources from environment variables
CAMERA_SOURCES = [
    os.getenv("CAM0", "0"),
    os.getenv("CAM1", None),
    os.getenv("CAM2", None),
    os.getenv("CAM3", None),
]

# Initialize available cameras only
def get_camera(source):
    """Safely initialize a camera with error handling."""
    if source is None:
        return None
    try:
        # Try to convert to int (for camera index), otherwise use as file path
        try:
            source_val = int(source)
        except (ValueError, TypeError):
            source_val = source
        
        cam = cv2.VideoCapture(source_val)
        if not cam.isOpened():
            return None
        return cam
    except Exception as e:
        print(f"Camera {source} failed: {e}")
        return None

cameras = {}
for i, source in enumerate(CAMERA_SOURCES):
    if source is not None:
        cam = get_camera(source)
        if cam:
            cameras[i] = cam
            print(f"✓ Camera {i} initialized from {source}")
        else:
            print(f"✗ Camera {i} failed to open: {source}")
    else:
        print(f"✗ Camera {i} not configured")

# Create snapshot directory
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Object detection count tracker
object_counts = Counter()

def gen_frames(camera, cam_id: int = 0):
    """Yield MJPEG frames from a given camera. Saves snapshots into per-camera subfolders."""
    last_snapshot_time = datetime.now()
    snapshot_interval = 5  # seconds

    # Ensure per-camera snapshot folder exists
    cam_str = str(cam_id)
    cam_snapshot_dir = SNAPSHOT_DIR / cam_str
    cam_snapshot_dir.mkdir(exist_ok=True)

    while True:
        success, frame = camera.read()
        if not success:
            # Yield a black frame with error message
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera Disconnected", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
                CONFIDENCE_THRESHOLD = 0.6
                for box in results.boxes:
                    confidence = float(box.conf[0])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    cls_id = int(box.cls[0])
                    class_name = results.names[cls_id]
                    detections[class_name] = True
                    object_counts[class_name] += 1

            # Store in DB
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
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/history.html", response_class=HTMLResponse)
async def history(request: Request):    
    return templates.TemplateResponse("history.html", {"request": request})

@app.get("/video")
def video():
    """Legacy/default video stream (primary camera)."""
    if 0 not in cameras:
        return HTMLResponse("Camera 0 not available", status_code=503)
    return StreamingResponse(
        gen_frames(cameras[0], 0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/video0")
def video0():
    """Stream from camera index 0."""
    if 0 not in cameras:
        return HTMLResponse("Camera 0 not available", status_code=503)
    return StreamingResponse(
        gen_frames(cameras[0], 0),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/video1")
def video1():
    """Stream from camera index 1."""
    if 1 not in cameras:
        return HTMLResponse("Camera 1 not available", status_code=503)
    return StreamingResponse(
        gen_frames(cameras[1], 1),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/video2")
def video2():
    """Stream from camera index 2."""
    if 2 not in cameras:
        return HTMLResponse("Camera 2 not available", status_code=503)
    return StreamingResponse(
        gen_frames(cameras[2], 2),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/video3")
def video3():
    """Stream from camera index 3."""
    if 3 not in cameras:
        return HTMLResponse("Camera 3 not available", status_code=503)
    return StreamingResponse(
        gen_frames(cameras[3], 3),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/counts")
async def get_counts():
    return JSONResponse(dict(object_counts))

@app.get("/get_snapshots")
async def get_snapshots(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format"),
    cam: Optional[str] = Query(None, description="Camera filter (e.g., '0' or '1')")
):
    return JSONResponse(get_all_snapshots(start_date, end_date, cam))

@app.get("/snapshots/{cam}/{filename}")
async def serve_snapshot(cam: str, filename: str):
    return FileResponse(SNAPSHOT_DIR / cam / filename)

@app.get('/available_cameras')
async def available_cameras():
    """Return a JSON list of available camera ids (connected cameras)."""
    return JSONResponse(sorted(list(cameras.keys())))
