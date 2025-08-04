import cv2
import numpy as np
from yolo_detector import YOLODetector
from sort import Sort

# Initialize YOLO and SORT
detector = YOLODetector()
tracker = Sort()

# Open webcam (0 = MacBook camera, 1 = iPhone Continuity Camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects: [x1, y1, x2, y2, conf, class_id]
    detections = detector.detect(frame, conf_thresh=0.25)

    # DEBUG: print detected class labels and confidence
    print(f"[DEBUG] Detections: {[(detector.class_names[int(d[5])], round(d[4], 2)) for d in detections]}")

    # Prepare detections for SORT: [x1, y1, x2, y2, conf]
    dets_for_sort = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        dets_for_sort.append([x1, y1, x2, y2, conf])

    dets_for_sort = np.array(dets_for_sort) if len(dets_for_sort) > 0 else np.empty((0, 5))

    # Run tracker
    tracks = tracker.update(dets_for_sort)

    # Draw tracking boxes with labels
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        # Try to match detection to get class label
        label = "Object"
        for det in detections:
            dx1, dy1, dx2, dy2, conf, cls_id = det
            if abs(dx1 - x1) < 20 and abs(dy1 - y1) < 20:
                label = f"{detector.class_names[int(cls_id)]}"
                break

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection & Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
