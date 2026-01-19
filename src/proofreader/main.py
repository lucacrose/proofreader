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
from .core.schema import ResolvedItem

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

    def _final_judge(self, item: ResolvedItem):
        if getattr(item, "_finalized", False):
            return
        
        # 1. Setup Evidence
        v_id = item.visual_id
        v_conf = item.visual_conf
        
        # Try direct mapping of OCR text to an ID
        ocr_name_raw = item.text_name.lower().strip()
        ocr_id_direct = self.matcher.name_to_id.get(ocr_name_raw)
        ocr_conf = item.text_conf / 100.0 if item.text_conf > 1 else item.text_conf

        # 2. AGREEMENT (The Easy Win)
        # If they agree on the ID, sum their confidences.
        if v_id != -1 and v_id == ocr_id_direct:
            item.id = v_id
            item.name = self.matcher.id_to_name.get(str(v_id))
            return

        # 3. HIGH CONFIDENCE INDIVIDUAL (The Specialist)
        # If CLIP is very strong (>0.85) OR OCR is very strong (>0.85)
        if v_conf > 0.85:
            item.id = v_id
            item.name = self.matcher.id_to_name.get(str(v_id))
            return
        
        if ocr_conf > 0.85 and ocr_id_direct:
            item.id = ocr_id_direct
            item.name = self.matcher.id_to_name.get(str(ocr_id_direct))
            return

        # 4. THE FUZZY FALLBACK (The Rescue)
        # If we get here, both signals are "blurry." We use fuzzy matching 
        # to see if the OCR text is just a typo of a known item.
        if len(ocr_name_raw) > 2:
            fuzzy_name = self.reader._fuzzy_match_name(ocr_name_raw)
            fuzzy_id = self.matcher.name_to_id.get(fuzzy_name.lower())
            
            if fuzzy_id:
                item.id = int(fuzzy_id)
                item.name = fuzzy_name
                return

        # 5. WEAK SIGNAL TIE-BREAKER
        # If fuzzy failed, just pick the strongest of the two weak signals
        if v_conf >= ocr_conf and v_id != -1:
            item.id = v_id
            item.name = self.matcher.id_to_name.get(str(v_id))
        elif ocr_id_direct:
            item.id = ocr_id_direct
            item.name = self.matcher.id_to_name.get(str(ocr_id_direct))
        else:
            item.id = 0
            item.name = "Unknown"

    def process_image(self, image_path: str, conf_threshold: float) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        boxes = self.detector.detect(image_path, conf_threshold)
        layout = self.resolver.resolve(boxes)
        image = cv2.imread(image_path)

        # Visual matching first
        self.matcher.match_item_visuals(image, layout)

        # Pre-promote strong visual matches
        for side in [layout.outgoing, layout.incoming]:
            for item in side.items:
                if item.visual_id != -1 and item.visual_conf >= 0.995:
                    item.id = item.visual_id
                    item.name = self.matcher.id_to_name.get(str(item.visual_id), "Unknown")
                    item._finalized = True  # mark as locked

        self.reader.process_layout(
            image,
            layout,
            skip_if=lambda item: getattr(item, "_finalized", False)
        )

        for side in [layout.outgoing, layout.incoming]:
            for item in side.items:
                self._final_judge(item)

        return layout.to_dict()
