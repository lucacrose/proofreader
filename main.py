import os
import torch
import json
from transformers import CLIPProcessor, CLIPModel
from core.detector import TradeDetector
from core.resolver import SpatialResolver
from core.ocr import OCRReader
from core.matcher import VisualMatcher

DB_PATH = "old/generator/assets/db.json"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

if os.path.exists(DB_PATH):
    with open(DB_PATH, "r") as f:
        ITEM_DATABASE = json.load(f)
else:
    raise FileNotFoundError(f"Could not find db.json at {DB_PATH}")

CACHE_FILE = "old/embedding_bank.pt"
if os.path.exists(CACHE_FILE):
    cache_data = torch.load(CACHE_FILE, weights_only=False)
    EMBEDDING_BANK = cache_data['embeddings']
else:
    raise FileNotFoundError("Embedding bank missing. Please run the builder to generate embedding_bank.pt")

detector = TradeDetector("assets/weights/yolo_v1.pt")
resolver = SpatialResolver()
reader = OCRReader()
matcher = VisualMatcher(embedding_bank=EMBEDDING_BANK, item_db=ITEM_DATABASE, clip_processor=clip_processor, clip_model=clip_model)

boxes = detector.detect("test.png")
trade_layout = resolver.resolve(boxes)

print(boxes)
print(trade_layout)

reader.process_layout("test.png", trade_layout)

print(trade_layout)

matcher.match_item_visuals("test.png", trade_layout)

print(trade_layout)
