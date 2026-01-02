import cv2
import os
import random
from pathlib import Path

def draw_labels(image_path, label_path):
    # Load the image
    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    
    # Class colors (BGR)
    colors = [
        (255, 0, 0),   # 0: Item Card (Blue)
        (0, 255, 0),   # 1: Thumb (Green)
        (0, 255, 255), # 2: Name (Yellow)
        (0, 0, 255),   # 3: Robux Line (Red)
        (255, 0, 255)  # 4: Value (Magenta)
    ]

    if not label_path.exists():
        return img

    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, nw, nh = map(float, line.split())
            
            # Convert YOLO format (0-1) back to pixel coordinates
            x1 = int((x - nw/2) * w)
            y1 = int((y - nh/2) * h)
            x2 = int((x + nw/2) * w)
            y2 = int((y + nh/2) * h)
            
            # Draw rectangle
            color = colors[int(cls)] if int(cls) < len(colors) else (255, 255, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"Class {int(cls)}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# Paths
img_dir = Path("generator/dataset/train/images")
lbl_dir = Path("generator/dataset/train/labels")

# Get random image
img_files = list(img_dir.glob("*.png"))
sample_img = random.choice(img_files)
sample_lbl = lbl_dir / f"{sample_img.stem}.txt"

# Show it
result = draw_labels(sample_img, sample_lbl)
cv2.imshow("Label Check", result)
cv2.waitKey(0)
cv2.destroyAllWindows()