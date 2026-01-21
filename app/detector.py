import requests
import base64
import cv2
import numpy as np
import os

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "ppe-fruwx-vgp3y/1")  # Default model

def detect_from_frame(frame):
    """
    Send frame to Roboflow API for detection
    """
    if not ROBOFLOW_API_KEY:
        return {"error": "ROBOFLOW_API_KEY not set", "predictions": []}
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Roboflow API endpoint
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": 40,  # Minimum confidence threshold (0-100)
        "overlap": 30      # NMS threshold
    }
    
    try:
        # Send request to Roboflow
        response = requests.post(
            url,
            params=params,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            return predictions
        else:
            print(f"Roboflow API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return {"error": f"API returned {response.status_code}", "predictions": []}
    
    except Exception as e:
        print(f"Exception during detection: {str(e)}")
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
