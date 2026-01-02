import cv2
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def check_for_double_boxes(results, class_id=2, threshold=0.3):
    """Returns True if any boxes of the same class overlap significantly."""
    boxes = [box.xyxy[0].cpu().numpy() for box in results.boxes if int(box.cls) == class_id]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > threshold:
                return True
    return False

# Configuration
model_path_new = "runs/detect/train49/weights/best.pt"
model_path_old = "runs/detect/train46/weights/best.pt"
model_new = YOLO(model_path_new)
model_old = YOLO(model_path_old)

folder = Path("downloader/media")
images = [f for f in folder.iterdir() if f.suffix in {".png", ".jpg", ".jpeg", ".webp"}]

print("Searching for overlapping 'Item Name' labels in EITHER model...")

found = False
while not found:
    test_image = random.choice(images)
    
    # Run prediction on both
    res_new = model_new.predict(source=test_image, conf=0.5, imgsz=640, verbose=False)[0]
    res_old = model_old.predict(source=test_image, conf=0.5, imgsz=640, verbose=False)[0]
    
    # Check if either model triggered the bug
    bug_new = check_for_double_boxes(res_new)
    bug_old = check_for_double_boxes(res_old)

    if bug_new or bug_old:
        status_msg = ""
        if bug_new and bug_old: status_msg = "BOTH models double-boxed!"
        elif bug_new: status_msg = "BUG in NEW model only!"
        else: status_msg = "BUG in OLD model only!"
        
        print(f"FOUND: {test_image.name} - {status_msg}")
        
        # Plot and Label
        img_n = res_new.plot()
        img_o = res_old.plot()
        
        # UI color coding: Red if buggy, Green if clean
        color_n = (0, 0, 255) if bug_new else (0, 255, 0)
        color_o = (0, 0, 255) if bug_old else (0, 255, 0)
        
        cv2.putText(img_o, "Old Model", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_o, 2)
        cv2.putText(img_n, "New Model", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_n, 2)
        
        # Layout and display
        combined = cv2.hconcat([img_o, img_n])
        h, w = combined.shape[:2]
        ratio = min(1600 / w, 900 / h)
        scaled = cv2.resize(combined, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_AREA)
        
        cv2.imshow("Double-Box Comparison (Press any key for next)", scaled)
        # Change waitKey(0) to found=True if you want it to stop at the first find
        if cv2.waitKey(0) == ord('q'):
            break

cv2.destroyAllWindows()