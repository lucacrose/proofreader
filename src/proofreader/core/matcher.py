import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any
from .schema import TradeLayout
from proofreader.core.config import VISUAL_MATCH_THRESHOLD

class VisualMatcher:
    def __init__(self, embedding_bank: Dict[str, np.ndarray], item_db: List[dict], clip_processor: Any, clip_model: Any, device: str = "cuda"):
        self.device = device
        self.bank = embedding_bank
        self.item_db = item_db
        self.clip_processor = clip_processor
        self.clip_model = clip_model

        self.name_to_id = {str(i["name"]).lower().strip(): i["id"] for i in item_db}
        self.id_to_name = {str(i["id"]): i["name"] for i in item_db}

        self.bank_names = list(embedding_bank.keys())
        self.bank_tensor = torch.stack([embedding_bank[name] for name in self.bank_names]).to(self.device)
        self.bank_tensor = torch.nn.functional.normalize(self.bank_tensor, dim=1)

    def _get_id_from_name(self, name: str) -> str:
        item = next((i for i in self.item_db if i["name"] == name), None)
        return item["id"] if item else 0

    def match_item_visuals(self, image: np.ndarray, layout: TradeLayout, similarity_threshold: float = VISUAL_MATCH_THRESHOLD):
        items_to_process = []
        crops = []
        
        for side in (layout.outgoing.items, layout.incoming.items):
            for item in side:
                if item.thumb_box:
                    x1, y1, x2, y2 = item.thumb_box.coords
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crops.append(pil_img)
                        items_to_process.append(item)

        if not crops:
            return
        
        inputs = self.clip_processor(images=crops, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            query_features = self.clip_model.get_image_features(**inputs)
            query_features = torch.nn.functional.normalize(query_features, dim=1)
            similarities = torch.matmul(query_features, self.bank_tensor.T)
            best_scores, best_indices = torch.max(similarities, dim=1)
        
        for i, item in enumerate(items_to_process):
            visual_match_val = self.bank_names[best_indices[i]]
            visual_conf = best_scores[i].item()

            is_ocr_valid = item.name.lower().strip() in self.name_to_id if item.name else False

            if (not is_ocr_valid or visual_conf > 0.95) and visual_conf >= similarity_threshold:
                if str(visual_match_val).isdigit():
                    item.id = int(visual_match_val)
                    item.name = self.id_to_name.get(str(visual_match_val), "Unknown Item")
                else:
                    item.name = visual_match_val
                    item.id = self._get_id_from_name(visual_match_val)
            else:
                item.id = self._get_id_from_name(item.name)
