from ultralytics import YOLO
import cv2

import random
from pathlib import Path

folder = Path("downloader/media")
images = [f for f in folder.iterdir() if f.suffix in {".png", ".jpg", ".jpeg", ".webp"}]

#downloader\media\916ff067-e76a-4853-8c6a-8978ea5015e1.png
#downloader\media\9d87905a-365a-4f2c-9441-ab4366b0bcb7.png
#downloader\media\83b6f192-e944-4e49-a862-4813a8b5301f.png nameless!
#downloader\media\4b6857de-218e-4cca-a4f1-cb22fa3dea06.png

#downloader\media\1dc5e56b-b456-46d4-80f2-7f37436f67a6.png
#downloader\media\b76a7856-9f65-46a6-b86a-a3031d4cdc0b.png

#downloader\media\d008b774-217f-4a8b-a11e-573c7b1602fa.png
#downloader\media\9dd56d30-fc47-4dfa-b968-27d941d0766d.png

#downloader\media\010fceb3-6c63-4777-ae2f-4c681ce2b02f.jpg
#downloader\media\b68295b2-7c80-4428-ba2c-d49a184c5a36.png
#downloader\media\5c0bca8a-4402-4ecd-b409-2fe3495fee6f.png
#downloader\media\4eaff886-250b-413c-bfbd-40b7d78f3c1d.png
#downloader\media\5283dd60-d3b6-4c7d-8bb4-a3039e80f040.png
#downloader\media\c1118483-5126-4bcf-8faa-cf2fde496740.jpg

#downloader\media\080622b6-0138-46ba-a1e1-26fd31d7e434.png
#downloader\media\1441eff2-d3bd-45ed-9e9b-e77772ee8dfc.png value non detect

#downloader\media\b88d7983-3b94-43a4-a218-aeb32c886234.png


# 2. Run the prediction
# 'conf=0.25' means only show things the AI is at least 25% sure about
from ultralytics import YOLO
import cv2
from pathlib import Path

while True:
    random_image = random.choice(images)

    print(random_image)

    # Paths to your models
    model_path_19 = "runs/detect/train28/weights/best.pt"#"runs/detect/train19/weights/best.pt"
    model_path_28 = "runs/detect/train49/weights/best.pt"#"runs/detect/train28/weights/best.pt"

    # Load both models
    model_19 = YOLO(model_path_19)
    model_28 = YOLO(model_path_28)

    # The image you want to test
    test_image = random_image#"downloader/media/9d87905a-365a-4f2c-9441-ab4366b0bcb7.png"

    # 1. Run inference on both (keep conf slightly higher to avoid clutter)
    res_19 = model_19.predict(source=test_image, conf=0.6, imgsz=640)[0]
    res_28 = model_28.predict(source=test_image, conf=0.6, imgsz=640)[0]

    # 2. Get the plotted images (the ones with boxes drawn)
    # We use .plot() to get the BGR array
    img_19 = res_19.plot()
    img_28 = res_28.plot()

    # 3. Add labels to the top of the images so you know which is which
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_19, "Model: Train46 (Old)", (20, 40), font, 1, (0, 255, 0), 2)
    cv2.putText(img_28, "Model: Train49 (New)", (20, 40), font, 1, (0, 255, 0), 2)

    # 4. Combine images side-by-side

    # 1. Stack side-by-side
    combined = cv2.hconcat([img_19, img_28])

    # 2. Get current dimensions
    h, w = combined.shape[:2]

    # 3. Define the max bounds for a 720p display
    # We use 1200x700 to leave a small margin for window borders/taskbar
    max_w, max_h = 1600, 900 

    # 4. Calculate the scaling ratio
    # This chooses the "strictest" limit to ensure the image never exceeds the bounds
    ratio = min(max_w / w, max_h / h)

    # 5. Calculate new dimensions
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # 6. Resize with high quality
    combined_scaled = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 7. Show the result
    cv2.imshow("Train19 vs Train28 - Aspect Ratio Maintained", combined_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()