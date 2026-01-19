import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from typing import Dict, List, Any
from .schema import TradeLayout

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
        
        # 1. Load the learned mapping from training
        with open(mapping_path, "r") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 2. Setup ID/Name lookups from your DB
        self.item_db = item_db
        self.id_to_name = {str(i["id"]): i["name"] for i in item_db}
        self.name_to_id = {str(i["name"]).lower().strip(): i["id"] for i in item_db}

        # 3. Load the Trained Model
        num_classes = len(self.class_to_idx)
        self.model = CLIPItemEmbedder(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 4. Pre-calculate the Normalized Bank (Prototypes)
        with torch.no_grad():
            self.bank_tensor = F.normalize(self.model.item_prototypes.weight, p=2, dim=-1)

        # 5. Training-matching transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def match_item_visuals(self, image: np.ndarray, layout: Any, similarity_threshold: float = 0.85):
        items_to_process = []
        crops = []
        
        # --- Existing Cropping Logic ---
        for side in (layout.outgoing.items, layout.incoming.items):
            for item in side:
                if item.thumb_box:
                    x1, y1, x2, y2 = item.thumb_box.coords
                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        # Convert CV2 image to PIL and Preprocess
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        processed_crop = self.preprocess(pil_img)
                        crops.append(processed_crop)
                        items_to_process.append(item)

        if not crops:
            return

        # Stack crops into a single batch [N, 3, 224, 224]
        batch_tensor = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            # Get features from the new vision encoder
            query_features = self.model(batch_tensor)
            
            # Calculate Cosine Similarities
            logits = query_features @ self.bank_tensor.t()
            
            # Apply our Confidence Scaling (The "MegaPhone")
            scaled_logits = logits * 100.0
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # Get the top match for each crop in the batch
            best_probs, best_indices = torch.max(probabilities, dim=1)
        
        # --- Update Layout Objects ---
        for i, item in enumerate(items_to_process):
            visual_idx = best_indices[i].item()
            visual_match_id = self.idx_to_class[visual_idx] # This is the Folder Name/ID
            visual_conf = best_probs[i].item()

            # Logic: If confidence is high (> 99%) or OCR failed, trust the Visual ID
            if visual_conf >= similarity_threshold:
                # Update the item object
                item.id = int(visual_match_id)
                item.name = self.id_to_name.get(str(visual_match_id), f"ID: {visual_match_id}")
                # Optional: store confidence for debugging
                item.visual_confidence = visual_conf 
            else:
                # Revert to OCR or mark as unknown if visual confidence is too low
                # This prevents "Dirt" being identified as a "Dominus"
                pass
