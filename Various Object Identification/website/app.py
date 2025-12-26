"""
Flask Web App for YOLOv8 Object Detection
Run: python app.py
Access: http://localhost:5000
"""

from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import math
import cvzone
import threading
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
               "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
               "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
               "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
               "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

# Global variables
cap = None
detection_active = False
current_frame = None
detection_stats = {"total_objects": 0, "last_detections": []}
lock = threading.Lock()

def initialize_camera():
    """Initialize camera capture"""
    global cap
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Test if camera is working
        ret, _ = cap.read()
        if not ret:
            print("[ERROR] Camera not accessible")
            return False
        print("[SUCCESS] Camera initialized")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return False

def detect_objects_thread():
    """Continuous object detection from camera"""
    global detection_active, current_frame, detection_stats, cap

    print("[INFO] Detection thread started")
    frame_count = 0

    while detection_active:
        try:
            if cap is None or not cap.isOpened():
                print("[WARNING] Camera not open, reconnecting...")
                time.sleep(1)
                continue

            success, img = cap.read()
            if not success:
                print("[WARNING] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Run YOLO detection
            results = model(img, verbose=False)
            current_detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    label = f'{class_names[cls]} {conf}'

                    # Draw bounding box with corner rect
                    try:
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, t=2, rt=1,
                                        colorC=(0, 255, 0), colorR=(255, 0, 0))
                        cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)),
                                         scale=0.7, thickness=1)
                    except Exception as e:
                        # Fallback to basic drawing
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (0, 255, 0), 2)

                    current_detections.append({
                        "class": class_names[cls],
                        "confidence": float(conf)
                    })

            # Update stats and frame
            with lock:
                detection_stats["total_objects"] = len(current_detections)
                detection_stats["last_detections"] = current_detections
                current_frame = img.copy()

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count} frames, Last detection: {len(current_detections)} objects")

        except Exception as e:
            print(f"[ERROR] Detection error: {e}")
            time.sleep(0.1)

def generate_frames():
    """Generate frames for streaming to web"""
    global current_frame, detection_active
    print("[INFO] Video stream generator started")

    frame_count = 0
    while detection_active:
        try:
            with lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("[WARNING] Failed to encode frame")
                time.sleep(0.01)
                continue

            frame_bytes = buffer.tobytes()

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n'
                   + frame_bytes + b'\r\n')

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[INFO] Streamed {frame_count} frames to browser")

            time.sleep(0.03)  # ~30 FPS

        except Exception as e:
            print(f"[ERROR] Stream generation error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/start')
def start_detection():
    """Start object detection"""
    global detection_active, cap

    if detection_active:
        return jsonify({"status": "info", "message": "Detection already running"})

    try:
        if not initialize_camera():
            return jsonify({"status": "error", "message": "Could not access camera"}), 500

        detection_active = True
        detection_thread = threading.Thread(target=detect_objects_thread, daemon=True)
        detection_thread.start()

        print("[INFO] Detection started")
        return jsonify({"status": "success", "message": "Detection started"})
    except Exception as e:
        print(f"[ERROR] Failed to start detection: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop')
def stop_detection():
    """Stop object detection"""
    global detection_active, cap, current_frame

    detection_active = False
    time.sleep(0.5)  # Give thread time to stop

    if cap is not None:
        cap.release()
        cap = None

    current_frame = None

    print("[INFO] Detection stopped")
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if not detection_active:
        return "Detection not running", 400

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    with lock:
        stats = {
            "total_objects": detection_stats["total_objects"],
            "last_detections": detection_stats["last_detections"]
        }
    return jsonify(stats)

@app.route('/classes')
def get_classes():
    """Get all available class names"""
    return jsonify({"classes": class_names})

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting Flask YOLOv8 Object Detection")
    print("=" * 50)
    print("🌐 Access the app at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000)
