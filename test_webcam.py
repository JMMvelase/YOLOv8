import cv2
import numpy as np
from dotenv import load_dotenv
from app.detector import detect_from_frame

# Load environment variables from .env file
load_dotenv()

# Create a simple test image
test_image = np.zeros((480, 640, 3), dtype=np.uint8)

# Test detection
result = detect_from_frame(test_image)

print("Detection result:")
print(result)