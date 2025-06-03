from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
from .detector import detect_from_frame  # <-- Your custom detector

app = FastAPI()

# Jinja2 will look inside 'app/templates'
templates = Jinja2Templates(directory="app/templates")

# Access the default webcam
camera = cv2.VideoCapture(0)

# Stream annotated frames using your YOLOv8 detector
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            annotated_frame, _ = detect_from_frame(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Render HTML page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Stream video to the frontend
@app.get("/video")
def video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
