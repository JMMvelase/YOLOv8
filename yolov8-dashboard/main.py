from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, Response
from datetime import datetime
from collections import Counte
import cv2
import os

from app.detector import detect_from_frame  # YOLO detection

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Initialize webcam
camera = cv2.VideoCapture(0)

# Snapshot saving setup
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Object detection count tracker
object_counts = Counter()

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        # YOLO detection + annotation
        annotated_frame, results = detect_from_frame(frame)

        # Count detected objects
        if results and results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = results.names[cls_id]
                object_counts[class_name] += 1

            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, annotated_frame)

        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
