import cv2
from ultralytics import YOLO

# Load an image from a specific path
image_path = 'object-recognition-using-python.jpg'  # Change this to the path of your image

img = cv2.imread(image_path)

model = YOLO("yolov8n.pt")

results = model(img)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
