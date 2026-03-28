import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# Load model once at startup
MODEL_PATH = Path(__file__).parent.parent / "weights" / "best.1.0.pt"
model = YOLO(str(MODEL_PATH))

def detect_from_frame(frame):
    """
    Run YOLOv8 inference on a frame using local weights
    """
    try:
        print(f"DEBUG: Running inference on frame with shape {frame.shape}")
        
        # Run inference
        results = model.track(frame, conf=0.7, persist=True)  # 0.7 = 70% confidence threshold
        
        predictions = []
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract detections
            if result.boxes is not None:
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                else:
                    track_ids = [None] * len(result.boxes)

                boxes = result.boxes.xyxy.cpu().numpy()  # ← fixed
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    predictions.append({
                        "x": float((x1 + x2) / 2),
                        "y": float((y1 + y2) / 2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "confidence": float(confs[i]),
                        "class": result.names[int(class_ids[i])],
                        "track_id": int(track_ids[i]) if track_ids[i] is not None else None
                    })
        print(f"DEBUG: Predictions: {len(predictions)} detections")
        return {"predictions": predictions}
    
    except Exception as e:
        print(f"Exception during detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "predictions": []}

def detect_from_upload(image_bytes):
    """
    Detect from uploaded image bytes
    """
    try:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Could not decode image", "predictions": []}
        
        return detect_from_frame(frame)
    
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return {"error": str(e), "predictions": []}
