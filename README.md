# 🧠 YOLOv8 Live Dashboard

A real-time object detection dashboard built with **YOLOv8**, **FastAPI**, and **OpenCV**. It streams a live webcam feed to a browser and displays detection counts of detected objects.

---

## 🚀 Features

- 🔴 Live MJPEG video stream from webcam
- 📦 Real-time object detection using YOLOv8
- 📊 Dynamic object count updates (auto-refresh every 2 seconds)
- 🖼️ Automatic snapshot saving when detections occur
- ⚙️ Easily extendable for analytics and alerts

---

## 📁 Project Structure

yolov8-dashboard/
├── app/
│ ├── templates/
│ │ └── index.html # Web UI
│ └── detector.py # YOLOv8 detection logic
├── snapshots/ # Saved detection snapshots
├── main.py # FastAPI backend
├── weights/
│ └── best.pt # Your custom-trained YOLOv8 model
├── test_webcam.py # Standalone OpenCV webcam tester
└── README.md # This file


---

## 🛠️ Setup Instructions

### 1. 🔁 Clone the repo


git clone https://github.com/your-username/yolov8-dashboard.git
cd yolov8-dashboard
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn opencv-python ultralytics jinja2 ''''

▶️ Run the App
uvicorn main:app --reload

🔍 Debugging Tips
test your webcam:
python test_webcam.py
