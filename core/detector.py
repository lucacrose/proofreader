from typing import List
from ultralytics.models import YOLO
from core.schema import Box

class TradeDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
        self.class_map = {
            0: "item_card",
            1: "item_thumb",
            2: "item_name",
            3: "robux_line",
            4: "robux_value"
        }

    def detect(self, image_source: str, conf_threshold: float) -> List[Box]:
        results = self.model.predict(image_source, conf=conf_threshold, verbose=False)[0]

        detected_boxes = []

        for box in results.boxes:
            coords = tuple(map(int, box.xyxy[0].tolist()))
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = self.class_map.get(cls_id, f"unknown_{cls_id}")

            detected_boxes.append(Box(
                coords=coords,
                label=label,
                confidence=conf
            ))

        return detected_boxes
