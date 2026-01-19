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
from .core.config import DB_PATH, MODEL_PATH, DEVICE, CLASS_MAP_PATH, CLIP_BEST_PATH, BASE_URL
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

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

        with open(DB_PATH, "r") as f:
            item_db = json.load(f)
        
        self.detector = TradeDetector(MODEL_PATH)
        self.resolver = SpatialResolver()
        self.reader = OCRReader(item_db)

        self.matcher = VisualMatcher(
            item_db=item_db,
            weights_path=CLIP_BEST_PATH,
            mapping_path=CLASS_MAP_PATH,
            device=self.device
        ) 

    def _ensure_assets(self):
        assets = {
            DB_PATH: f"{BASE_URL}/item_database.json",
            MODEL_PATH: f"{BASE_URL}/yolo.pt",
            CLIP_BEST_PATH: f"{BASE_URL}/clip.pt",
            CLASS_MAP_PATH: f"{BASE_URL}/class_mapping.json"
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
        
        v_id = item.visual_id
        v_conf = item.visual_conf

        ocr_name_raw = item.text_name.lower().strip()
        ocr_id_direct = self.matcher.name_to_id.get(ocr_name_raw)
        ocr_conf = item.text_conf / 100.0 if item.text_conf > 1 else item.text_conf

        if v_id != -1 and v_id == ocr_id_direct:
            item.id = v_id
            item.name = self.matcher.id_to_name.get(str(v_id))
            return
        
        if v_conf > 0.85:
            item.id = v_id
            item.name = self.matcher.id_to_name.get(str(v_id))
            return
        
        if ocr_conf > 0.85 and ocr_id_direct:
            item.id = ocr_id_direct
            item.name = self.matcher.id_to_name.get(str(ocr_id_direct))
            return
        
        if len(ocr_name_raw) > 2:
            fuzzy_name = self.reader._fuzzy_match_name(ocr_name_raw)
            fuzzy_id = self.matcher.name_to_id.get(fuzzy_name.lower())
            
            if fuzzy_id:
                item.id = int(fuzzy_id)
                item.name = fuzzy_name
                return
        
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

        self.matcher.match_item_visuals(image, layout)

        for side in [layout.outgoing, layout.incoming]:
            for item in side.items:
                if item.visual_id != -1 and item.visual_conf >= 0.995:
                    item.id = item.visual_id
                    item.name = self.matcher.id_to_name.get(str(item.visual_id), "Unknown")
                    item._finalized = True

        self.reader.process_layout(
            image,
            layout,
            skip_if=lambda item: getattr(item, "_finalized", False)
        )

        for side in [layout.outgoing, layout.incoming]:
            for item in side.items:
                self._final_judge(item)

        return layout.to_dict()
