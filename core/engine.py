import os
import cv2
import torch
import json
from transformers import CLIPProcessor, CLIPModel
from core.detector import TradeDetector
from core.resolver import SpatialResolver
from core.ocr import OCRReader
from core.matcher import VisualMatcher

class TradeEngine:
    def __init__(self, model_path="assets/weights/yolo_v1.pt", db_path="assets/db.json", cache_file="assets/embedding_bank.pt", device: str = None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        with open(db_path, "r") as f:
            item_db = json.load(f)
        
        cache_data = torch.load(cache_file, weights_only=False)['embeddings']
        self.embeddings = {k: torch.tensor(v).to(self.device) for k, v in cache_data.items()}

        self.detector = TradeDetector(model_path)
        self.resolver = SpatialResolver()
        
        self.reader = OCRReader(gpu=(self.device == "cuda"))
        
        self.matcher = VisualMatcher(
            embedding_bank=self.embeddings,
            item_db=item_db,
            clip_processor=self.clip_processor,
            clip_model=self.clip_model,
            device=self.device
        )

    def process_image(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        boxes = self.detector.detect(image_path)
        layout = self.resolver.resolve(boxes)

        image = cv2.imread(image_path)

        self.reader.process_layout(image, layout)

        self.matcher.match_item_visuals(image, layout)

        return layout.to_dict()
