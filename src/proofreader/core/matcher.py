import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from typing import List
from .schema import TradeLayout, ResolvedItem

class CLIPItemEmbedder(nn.Module):
    def __init__(self, num_classes, model_id="openai/clip-vit-base-patch32"):
        super().__init__()
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.item_prototypes = nn.Embedding(num_classes, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        return F.normalize(outputs.image_embeds, p=2, dim=-1)

class VisualMatcher:
    def __init__(self, weights_path: str, mapping_path: str, item_db: List[dict], device: str = "cuda"):
        self.device = device

        with open(mapping_path, "r") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.id_to_name = {str(i["id"]): i["name"] for i in item_db}
        self.name_to_id = {str(i["name"]).lower().strip(): i["id"] for i in item_db}

        num_classes = len(self.class_to_idx)
        self.model = CLIPItemEmbedder(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        with torch.inference_mode():
            self.bank_tensor = F.normalize(self.model.item_prototypes.weight, p=2, dim=-1)
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def match_item_visuals(self, image: np.ndarray, layout: TradeLayout):
        items_to_process: List[ResolvedItem] = []
        crops = []

        for side in (layout.outgoing.items, layout.incoming.items):
            for item in side:
                if item.thumb_box:
                    x1, y1, x2, y2 = item.thumb_box.coords
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        processed_crop = self.preprocess(pil_img)
                        crops.append(processed_crop)
                        items_to_process.append(item)

        if not crops:
            return
        
        batch_tensor = torch.stack(crops).to(self.device)
        
        with torch.inference_mode():
            query_features = self.model(batch_tensor)

            logits = query_features @ self.bank_tensor.t()

            probabilities = F.softmax(logits * 100.0, dim=-1)
            best_probs, best_indices = torch.max(probabilities, dim=1)
        
        for i, item in enumerate(items_to_process):
            visual_idx = best_indices[i].item()

            visual_match_id_str = self.idx_to_class[visual_idx]

            item.visual_id = int(visual_match_id_str)
            item.visual_conf = float(best_probs[i].item())
