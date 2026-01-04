import os
import cv2
import torch
import json
from .detector import TradeDetector
from .resolver import SpatialResolver
from .ocr import OCRReader
from .matcher import VisualMatcher
from .config import DB_PATH, CACHE_PATH, MODEL_PATH, DEVICE

class TradeEngine:
    def __init__(self):
        self.device = DEVICE
        
        with open(DB_PATH, "r") as f:
            item_db = json.load(f)
        
        cache_data = torch.load(CACHE_PATH, weights_only=False)['embeddings']
        self.embeddings = {k: torch.tensor(v).to(self.device) for k, v in cache_data.items()}

        self.detector = TradeDetector(MODEL_PATH)
        self.resolver = SpatialResolver()
        
        self.reader = OCRReader(gpu=(self.device == "cuda"))
        
        self.matcher = VisualMatcher(
            embedding_bank=self.embeddings,
            item_db=item_db,
            clip_processor=self.clip_processor,
            clip_model=self.clip_model,
            device=self.device
        )

    def process_image(self, image_path: str, conf_threshold: float) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        boxes = self.detector.detect(image_path, conf_threshold)
        layout = self.resolver.resolve(boxes)

        image = cv2.imread(image_path)

        self.reader.process_layout(image, layout)

        self.matcher.match_item_visuals(image, layout)

        return layout.to_dict()

def get_trade_data(image_path: str, conf_threshold: float = 0.25):
    engine = TradeEngine()
    return engine.process_image(image_path, conf_threshold)
