from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import os
import time
import threading
from datetime import datetime
from collections import Counter

from app.detector import detect_from_frame

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

camera = cv2.VideoCapture(0)
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

object_counts = Counter()

def auto_clear_snapshots(folder=SNAPSHOT_DIR, max_age_minutes=2):
    while True:
        now = time.time()
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_minutes * 60:
                    os.remove(filepath)
                    print(f"Deleted old snapshot: {filename}")
        time.sleep(60)

# Start the cleanup thread
threading.Thread(target=auto_clear_snapshots, daemon=True).start()

def gen_frames():
    global object_counts
    while True:
        success, frame = camera.read()
        if not success:
            break

        annotated_frame, results = detect_from_frame(frame)

        # Reset counts for this frame
        object_counts = Counter()

        # Count detected objects
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())  # .item() ensures it's a native int
                class_name = results.names[cls_id]
                object_counts[class_name] += 1

        # Save snapshot if objects were detected
       # Save snapshot only if confidence > 0.80 for any detected object
    # Define target classes
    PPE_CLASSES = {"helmet", "vest", "gloves", "mask"}  # You can add more class names here
    CONFIDENCE_THRESHOLD = 0.80
    
    # Save snapshot only if a high-confidence target class is detected
    high_confidence_detected = any(
        float(box.conf[0]) > CONFIDENCE_THRESHOLD and 
        results.names[int(box.cls[0])] in PPE_CLASSES
        for box in results.boxes
    )
    
    if high_confidence_detected:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.jpg")
        cv2.imwrite(snapshot_path, annotated_frame)
    
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
    
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/counts")
async def get_counts():
    return JSONResponse(content=dict(object_counts))
