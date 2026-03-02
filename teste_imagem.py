from ultralytics import YOLO
import os
import cv2

model = YOLO("yolov8n.pt")
results = model("foto.jpg")

for r in results:
    img