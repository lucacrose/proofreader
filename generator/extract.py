import easyocr
from ultralytics import YOLO
import cv2
from thefuzz import process, fuzz
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import warnings

# --- CONFIGURATION & INITIALIZATION ---
# Suppress the 'use_fast' and FutureWarnings from transformers/torch
warnings.filterwarnings("ignore", category=FutureWarning)

model = YOLO("runs/detect/train49/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# CLIP Setup - Explicitly set use_fast=True to avoid the warning
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

CACHE_FILE = "embedding_bank.pt"
ITEM_DB_PATH = "generator/assets/db.json"
IMAGE_ASSETS_DIR = "generator/assets/images"

EMBEDDING_BANK = {}
ITEM_DATABASE = []

def get_clip_embedding(pil_img):
    """Generates a normalized vector representing the image."""
    inputs = clip_processor(images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.numpy().flatten()

# --- CACHE MANAGEMENT ---
rebuild_needed = False

# Check if DB has been updated since the last cache was built
if os.path.exists(CACHE_FILE) and os.path.exists(ITEM_DB_PATH):
    cache_mtime = os.path.getmtime(CACHE_FILE)
    db_mtime = os.path.getmtime(ITEM_DB_PATH)
    if db_mtime > cache_mtime:
        print("Detected updated db.json. Forcing rebuild of embedding bank...")
        rebuild_needed = True

if os.path.exists(CACHE_FILE) and not rebuild_needed:
    print(f"Loading Embedding Bank from {CACHE_FILE}...")
    try:
        # weights_only=False is required for PyTorch 2.6+ to load NumPy arrays in dicts
        cache_data = torch.load(CACHE_FILE, weights_only=False)
        EMBEDDING_BANK = cache_data['embeddings']
        ITEM_DATABASE = cache_data['names']
        print(f"Successfully loaded {len(ITEM_DATABASE)} items from cache.")
    except Exception as e:
        print(f"Cache load failed ({e}). Rebuilding...")
        rebuild_needed = True

if not EMBEDDING_BANK or rebuild_needed:
    print("Building Visual Embedding Bank (this may take a minute)...")
    if not os.path.exists(ITEM_DB_PATH):
        raise FileNotFoundError(f"Could not find db.json at {ITEM_DB_PATH}")
        
    with open(ITEM_DB_PATH, "r") as f:
        items = json.load(f)
        
    # Reset databases for rebuild
    EMBEDDING_BANK = {}
    ITEM_DATABASE = []
    
    for filename in os.listdir(IMAGE_ASSETS_DIR):
        item_id = os.path.splitext(filename)[0]
        # Match filename ID to name in JSON
        item_info = next((i for i in items if i["id"] == item_id), None)
        if item_info:
            name = item_info["name"]
            ITEM_DATABASE.append(name)
            img_path = os.path.join(IMAGE_ASSETS_DIR, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                EMBEDDING_BANK[name] = get_clip_embedding(img)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    
    torch.save({'embeddings': EMBEDDING_BANK, 'names': ITEM_DATABASE}, CACHE_FILE)
    print(f"Bank built and cached with {len(ITEM_DATABASE)} items!")

# --- HELPER FUNCTIONS ---
def is_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1-10 and iy1 >= oy1-10 and ix2 <= ox2+10 and iy2 <= oy2+10

def robust_visual_check(crop_cv2, ocr_hint=None):
    """Checks image against bank, prioritizing the hint provided by OCR."""
    crop_rgb = cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB)
    h, w = crop_rgb.shape[:2]
    
    # Trim 8% from edges to remove YOLO bounding box lines/artifacts
    m = 0.08 
    crop_rgb = crop_rgb[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
    
    crop_pil = Image.fromarray(crop_rgb)
    query_vec = get_clip_embedding(crop_pil)
    
    # 1. Quick Validation: Check if the OCR name is "visually plausible"
    if ocr_hint in EMBEDDING_BANK:
        hint_dist = cosine(query_vec, EMBEDDING_BANK[ocr_hint])
        # If the visual match for the OCR word is strong, return immediately
        if hint_dist < 0.22:
            return ocr_hint, hint_dist, []

    # 2. Comprehensive Search: Find the closest 3 items in the entire bank
    all_matches = []
    for item_name, ref_vec in EMBEDDING_BANK.items():
        dist = cosine(query_vec, ref_vec)
        all_matches.append((item_name, dist))
            
    all_matches.sort(key=lambda x: x[1]) # Lowest distance first
    top_3 = all_matches[:3]
    
    return top_3[0][0], top_3[0][1], top_3

# --- MAIN PROCESSING ---
def process_screenshot(img_path):
    img_name = Path(img_path).stem
    results = model.predict(img_path, conf=0.5, verbose=False)[0]
    orig_img = results.orig_img
    
    # Categorize YOLO detections
    boxes = results.boxes
    cards, thumbs, names = [], [], []
    for box in boxes:
        cls = int(box.cls)
        coords = list(map(int, box.xyxy[0]))
        data = {"coords": coords, "conf": float(box.conf)}
        if cls == 0: cards.append(data)
        elif cls == 1: thumbs.append(data)
        elif cls == 2: names.append(data)

    print(f"\n--- Results for {img_name} ---")
    
    for i, card in enumerate(cards):
        c_box = card["coords"]
        # Find thumb and name label inside this specific card
        my_thumb = [t for t in thumbs if is_inside(t["coords"], c_box)]
        my_names = [n for n in names if is_inside(n["coords"], c_box)]
        
        best_ocr, ocr_score = "N/A", 0
        visual_name, v_dist, top_matches = "N/A", 1.0, []

        # 1. Handle OCR
        if my_names:
            target_name = sorted(my_names, key=lambda x: x["conf"], reverse=True)[0]
            nx1, ny1, nx2, ny2 = target_name['coords']
            name_crop = orig_img[ny1:ny2, nx1:nx2]
            # Convert to gray for better OCR
            gray_crop = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
            raw_text = " ".join([t[1] for t in reader.readtext(gray_crop)])
            
            # Fuzzy match the raw text against our actual item database
            match_res = process.extractOne(raw_text, ITEM_DATABASE, scorer=fuzz.token_sort_ratio)
            if match_res:
                best_ocr, ocr_score = match_res[0], match_res[1]

        # 2. Handle Visual Similarity
        if my_thumb:
            tx1, ty1, tx2, ty2 = my_thumb[0]["coords"]
            visual_name, v_dist, top_matches = robust_visual_check(orig_img[ty1:ty2, tx1:tx2], ocr_hint=best_ocr)

        # 3. Decision Logic (Validation)
        status = "⚠️"
        # If OCR and Visual exactly agree
        if best_ocr == visual_name:
            status = "✅"
        # If OCR is near perfect, allow a slightly higher visual distance (Forced Match)
        elif ocr_score >= 95 and v_dist < 0.28:
            status = "✅ (Forced)"
            visual_name = best_ocr
        # If OCR is decent and it's in the visual top 3
        elif ocr_score > 80 and any(m[0] == best_ocr for m in top_matches):
            status = "✅ (Top-3)"
            visual_name = best_ocr
        # FALLBACK: If visual match shares a strong similarity in name with OCR (Helps with headphones/piephones)
        elif ocr_score > 75 and visual_name != "N/A":
            name_sim = fuzz.partial_ratio(best_ocr.lower(), visual_name.lower())
            if name_sim > 80 and v_dist < 0.25:
                status = "✅ (Fuzzy-Visual)"
                visual_name = f"{best_ocr} (via {visual_name})"

        print(f"{status} Card {i}: OCR='{best_ocr}' ({ocr_score}%) | Visual='{visual_name}' (Dist: {v_dist:.3f})")
        
        # Help diagnose why the warning triggered
        if status == "⚠️" and top_matches:
            debug_str = ", ".join([f"{name} ({d:.2f})" for name, d in top_matches])
            print(f"   ∟ Debug Top 3 Visuals: {debug_str}")

# Run analysis
process_screenshot("downloader/media/916ff067-e76a-4853-8c6a-8978ea5015e1.png")