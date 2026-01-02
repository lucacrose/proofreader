import torch
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cosine
from typing import Dict, List, Any
from core.schema import TradeLayout

class VisualMatcher:
    def __init__(self, embedding_bank: Dict[str, np.ndarray], item_db: List[dict], clip_processor: Any, clip_model: Any):
        self.bank = embedding_bank
        self.item_db = item_db
        self.clip_processor = clip_processor
        self.clip_model = clip_model

    def _get_id_from_name(self, name: str) -> str:
        if not self.item_db:
            return 0
        item = next((i for i in self.item_db if i["name"] == name), None)
        return int(item["id"]) if item else 0

    def match_item_visuals(self, image_source: str, layout: TradeLayout):
        image = cv2.imread(image_source)

        for side in (layout.outgoing.items, layout.incoming.items):
            for item in side:
                if not item.thumb_box:
                    continue
                
                x1, y1, x2, y2 = item.thumb_box.coords
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                inputs = self.clip_processor(images=pil_img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    query_features = self.clip_model.get_image_features(**inputs)
                    query_vec = query_features.numpy().flatten()
                
                best_match = item.name
                min_dist = 1.0

                if item.name in self.bank:
                    hint_dist = cosine(query_vec, self.bank[item.name])
                    if hint_dist < 0.2:
                        item.id = self._get_id_from_name(item.name)
                        continue
                
                for name, ref_vec in self.bank.items():
                    dist = cosine(query_vec, ref_vec)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = name
                
                item.name = best_match
                item.id = self._get_id_from_name(best_match)
