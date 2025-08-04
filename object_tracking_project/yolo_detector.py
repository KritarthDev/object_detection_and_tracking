from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")   # Auto-downloads
        self.model.to("cpu")              # Force CPU to avoid device issues
        self.class_names = self.model.model.names

    def detect(self, frame, conf_thresh=0.25):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls_id])
        return detections
