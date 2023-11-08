from flask import Flask, render_template, request
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    model = YOLO("yolov8n.pt")

    results = model(image)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite('output_image.jpg', image)

    return 'output_image.jpg'


if __name__ == '__main__':
    app.run(debug=True)
