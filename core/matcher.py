import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any
from core.schema import TradeLayout

class VisualMatcher:
    def __init__(self, embedding_bank: Dict[str, np.ndarray], item_db: List[dict], clip_processor: Any, clip_model: Any, device: str = "cuda"):
        self.device = device
        self.bank = embedding_bank
        self.item_db = item_db
        self.clip_processor = clip_processor
        self.clip_model = clip_model

        self.bank_names = list(embedding_bank.keys())
        self.bank_tensor = torch.stack([embedding_bank[name] for name in self.bank_names]).to(self.device)
        self.bank_tensor = torch.nn.functional.normalize(self.bank_tensor, dim=1)

    def _get_id_from_name(self, name: str) -> str:
        item = next((i for i in self.item_db if i["name"] == name), None)
        return int(item["id"]) if item else 0

    def match_item_visuals(self, image: np.ndarray, layout: TradeLayout):
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

            best_indices = torch.argmax(similarities, dim=1)
        
        for i, item in enumerate(items_to_process):
            name = self.bank_names[best_indices[i]]
            item.name = name
            item.id = self._get_id_from_name(name)
