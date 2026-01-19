import os
import cv2
import torch
import json
import requests
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from .core.detector import TradeDetector
from .core.resolver import SpatialResolver
from .core.ocr import OCRReader
from .core.matcher import VisualMatcher
from .core.config import DB_PATH, MODEL_PATH, DEVICE, CLASS_MAP_PATH, CLIP_BEST_PATH

class TradeEngine:
    def __init__(self):
        self._ensure_assets()

        if DEVICE == "cpu" and not torch.cuda.is_available():
            import subprocess
            try:
                subprocess.check_output('nvidia-smi')
                print("Detected NVIDIA GPU, but your current Torch installation is CPU-only.")
                print("To fix this, run: pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            except:
                pass

        self.device = DEVICE

        # Initialize Base CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # Load Data
        with open(DB_PATH, "r") as f:
            item_db = json.load(f)

        with open(CLASS_MAP_PATH, "r") as f:
            class_map = json.load(f)

        # Initialize Components
        self.detector = TradeDetector(MODEL_PATH)
        self.resolver = SpatialResolver()
        self.reader = OCRReader(item_db)
        
        # Initialize Matcher (Prototypes are loaded internally from CLIP_BEST_PATH)
        self.matcher = VisualMatcher(
            item_db=item_db,
            #clip_processor=self.clip_processor,
            #clip_model=self.clip_model,
            weights_path=CLIP_BEST_PATH,
            mapping_path=CLASS_MAP_PATH,
            device=self.device
        ) 

    def _ensure_assets(self):
        BASE_URL = "https://github.com/lucacrose/proofreader/releases/latest/download"
        
        assets = {
            DB_PATH: f"{BASE_URL}/db.json",
            MODEL_PATH: f"{BASE_URL}/yolo.pt",
            CLIP_BEST_PATH: f"{BASE_URL}/item_clip_best.pt" # Ensure this is in your assets
        }

        for path, url in assets.items():
            if not path.exists():
                print(f"📦 {path.name} missing. Downloading from latest release...")
                self._download_file(url, path)

    def _download_file(self, url, dest_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=dest_path.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    def process_image(self, image_path: str, conf_threshold: float) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        boxes = self.detector.detect(image_path, conf_threshold)
        layout = self.resolver.resolve(boxes)
        image = cv2.imread(image_path)

        self.reader.process_layout(image, layout)
        self.matcher.match_item_visuals(image, layout)

        return layout.to_dict()