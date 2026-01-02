import os
import torch
import json
from transformers import CLIPProcessor, CLIPModel
from core.detector import TradeDetector
from core.resolver import SpatialResolver
from core.ocr import OCRReader
from core.matcher import VisualMatcher

class TradeEngine:
    def __init__(self, model_path="assets/weights/yolo_v1.pt", db_path="assets/db.json", cache_file="assets/embedding_bank.pt"):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        with open(db_path, "r") as f:
            item_db = json.load(f)
            
        embeddings = torch.load(cache_file, weights_only=False)['embeddings']

        self.detector = TradeDetector(model_path)
        self.resolver = SpatialResolver()
        self.reader = OCRReader()
        self.matcher = VisualMatcher(
            embedding_bank=embeddings, 
            item_db=item_db, 
            clip_processor=self.clip_processor, 
            clip_model=self.clip_model
        )

    def process_image(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        boxes = self.detector.detect(image_path)
        layout = self.resolver.resolve(boxes)

        self.reader.process_layout(image_path, layout)

        self.matcher.match_item_visuals(image_path, layout)

        return layout.to_dict()
