from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
result = model("images", show=True)
cv2.waitKey(0)
