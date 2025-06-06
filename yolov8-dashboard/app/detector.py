from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")  # Your trained model

def detect_from_frame(frame):
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw boxes
    return annotated_frame, results[0]
