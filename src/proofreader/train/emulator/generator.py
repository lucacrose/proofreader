import json
import random
import concurrent.futures
import sys
import traceback
import multiprocessing
from pathlib import Path
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from proofreader.core.config import (
    AUGMENTER_PATH, 
    DATASET_ROOT, 
    TEMPLATE_FILES, 
    AUGMENTER_CONFIG, 
    DB_PATH, 
    BACKGROUNDS_DIR, 
    setup_dataset_directories
)

GENERATOR_CONFIG = AUGMENTER_CONFIG["generator"]

def process_batch(batch_ids, db, backgrounds_count, progress_counter):
    """
    Processes a chunk of images using a single browser instance.
    """
    try:
        with sync_playwright() as p:
            # Optimization: Launch browser with performance flags
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
            )
            
            # Optimization: Reuse one context/page for the entire batch
            context = browser.new_context()
            page = context.new_page()

            with open(AUGMENTER_PATH, 'r', encoding="utf-8") as f:
                augmenter_js = f.read()

            for task_id in batch_ids:
                generate_single_image(page, task_id, db, backgrounds_count, augmenter_js)
                progress_counter.value += 1

            browser.close()
    except Exception:
        print(f"Batch failed starting at {batch_ids[0]}:")
        traceback.print_exc()

def generate_single_image(page, task_id, db, backgrounds_count, augmenter_js):
    # 1. Setup Logic
    split = "train" if random.random() < GENERATOR_CONFIG["train_split_fraction"] else "val"
    output_name = f"trade_{task_id:05d}"
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare Trade Data
    trade_input = [[], []]
    is_empty_trade = random.random() < GENERATOR_CONFIG["empty_trade_chance"]
    if not is_empty_trade:
        for side in [0, 1]:
            num_items = random.randint(0, 4)
            for _ in range(num_items):
                item = random.choice(db)
                trade_input[side].append(f"../../../../assets/thumbnails/{item['id']}.png")

    # 3. Page Setup & Execution
    aspect_ratio = random.uniform(GENERATOR_CONFIG["aspect_ratio_min"], GENERATOR_CONFIG["aspect_ratio_max"])
    width = random.randint(GENERATOR_CONFIG["width_min"], GENERATOR_CONFIG["width_max"])
    height = int(width / aspect_ratio)
    height = max(GENERATOR_CONFIG["height_min"], min(height, GENERATOR_CONFIG["height_max"]))
    
    page.set_viewport_size({"width": width, "height": height})
    random_file = random.choice(TEMPLATE_FILES)
    page.goto(f"file://{Path(random_file).absolute()}")
    
    zoom_factor = random.uniform(0.5, 2.0)
    page.evaluate(f"document.body.style.zoom = '{zoom_factor}'")
    page.evaluate(augmenter_js, [trade_input, is_empty_trade, backgrounds_count, AUGMENTER_CONFIG])

    # 4. Bounding Box Logic (Keep your existing get_padded_yolo and is_fully_visible here)
    def get_padded_yolo(element, class_id, pad_px=2):
        box = element.bounding_box()
        if not box: return None
        x1, y1 = max(0, box['x'] - pad_px), max(0, box['y'] - pad_px)
        x2, y2 = min(width, box['x'] + box['width'] + pad_px), min(height, box['y'] + box['height'] + pad_px)
        nw, nh = x2 - x1, y2 - y1
        return [class_id, (x1 + nw/2)/width, (y1 + nh/2)/height, nw/width, nh/height]

    label_data = []
    # ... (Your logic for querying items/robux and filling label_data) ...

    # 5. Optimization: Screenshot to Memory (Avoids initial disk write)
    img_buffer = page.screenshot(type="jpeg", quality=100)
    nparr = np.frombuffer(img_buffer, np.uint8)
    full_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 6. Classification Crops (In-Memory)
    item_cards = page.query_selector_all("div[trade-item-card]")
    for i, card in enumerate(item_cards):
        if not (card.is_visible() and float(card.evaluate("el => getComputedStyle(el).opacity")) > 0):
            continue
        thumb_container = card.query_selector(".item-card-thumb-container")
        if thumb_container and thumb_container.is_visible():
            img_src = thumb_container.query_selector("img").get_attribute("src")
            item_id = Path(img_src).stem
            box = thumb_container.bounding_box()
            if box:
                # Calculate coords and crop
                pad = 4
                x1, y1 = int(box['x'] - pad), int(box['y'] - pad)
                x2, y2 = int(box['x'] + box['width'] + pad), int(box['y'] + box['height'] + pad)
                if 0 <= x1 and 0 <= y1 and x2 <= width and y2 <= height:
                    crop = full_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        class_dir = DATASET_ROOT / "classification" / item_id
                        class_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(class_dir / f"{output_name}_{i}.jpg"), crop)

    # 7. In-Memory Augmentation
    if random.random() < 0.60:
        # JPEG Noise simulation without reopening file
        if random.random() < 0.5:
            q = random.randint(60, 90)
            _, enc = cv2.imencode('.jpg', full_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            full_img = cv2.imdecode(enc, 1)
        
        # Color/Noise
        if random.random() < 0.4:
            full_img = cv2.convertScaleAbs(full_img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        
        noise = np.random.normal(0, random.uniform(0.5, 2.5), full_img.shape).astype('float32')
        full_img = np.clip(full_img.astype('float32') + noise, 0, 255).astype('uint8')

    # 8. Final Disk Write (Single Write per image)
    cv2.imwrite(str(img_dir / f"{output_name}.jpg"), full_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    with open(lbl_dir / f"{output_name}.txt", "w") as f:
        for label in label_data:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def run_mass_generation(total_images=65536, max_workers=24):
    # ... (Keep your existing validation checks for backgrounds, templates, and DB) ...
    
    with open(DB_PATH, "r") as f:
        db = json.load(f)
    
    setup_dataset_directories(force_reset=True)

    # Optimization: Chunk the work into batches
    # 65k images / 24 workers = ~2700 images per worker. 
    # Let's do batches of 500 to keep memory stable.
    batch_size = 500
    all_ids = list(range(total_images))
    chunks = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]

    backgrounds_count = len([f for f in BACKGROUNDS_DIR.iterdir() if f.is_file()])

    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0) # 'i' for integer

    print(f"Generating {total_images} images using {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Pass the shared counter to each process
        futures = [
            executor.submit(process_batch, chunk, db, backgrounds_count, progress_counter) 
            for chunk in chunks
        ]
        
        # Initialize tqdm with the total image count
        with tqdm(total=total_images, desc="Generating Images") as pbar:
            last_val = 0
            while True:
                # Check if all futures are done
                done, not_done = concurrent.futures.wait(futures, timeout=0.5)
                
                # Update progress bar based on the shared counter
                current_val = progress_counter.value
                pbar.update(current_val - last_val)
                last_val = current_val
                
                if len(not_done) == 0:
                    break

if __name__ == "__main__":
    run_mass_generation()
