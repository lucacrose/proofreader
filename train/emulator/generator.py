import json
import random
import string
import concurrent.futures
import sys
import traceback
from pathlib import Path
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.config import (
    TEMPLATES_DIR, 
    AUGMENTER_PATH, 
    DATASET_ROOT, 
    TEMPLATE_FILES, 
    setup_dataset_directories
)

def generate_random_username():
    length = random.randint(3, 20)
    chars = string.ascii_letters + string.digits + "_"
    return "".join(random.choice(chars) for _ in range(length))

def worker_task(task_id, db, backgrounds_count):
    try:
        split = "train" if random.random() < 0.8 else "val"
        output_name = f"trade_{task_id:05d}"

        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        trade_input = [[], []]
        is_empty_trade = random.random() < 0.09

        if not is_empty_trade:
            for side in [0, 1]:
                num_items = random.randint(0, 4)
                for _ in range(num_items):
                    item = random.choice(db)
                    trade_input[side].append({
                        "id": item["id"],
                        "name": item["name"],
                        "img": f"../../../assets/thumbnails/{item['id']}.png"
                    })
        
        with open(AUGMENTER_PATH, 'r', encoding="utf-8") as f:
            augmenter_js = f.read()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            aspect_ratio = random.uniform(1.0, 2.4)
            width = random.randint(800, 2560)
            height = int(width / aspect_ratio)
            height = max(400, min(height, 1600))
            
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()

            random_file = random.choice(TEMPLATE_FILES)
            page.goto(f"file://{Path(random_file).absolute()}")

            user_meta = {"display": generate_random_username(), "handle": generate_random_username()}
            page.evaluate(augmenter_js, [trade_input, user_meta, is_empty_trade, backgrounds_count])

            def get_padded_yolo(element, class_id, pad_px=2):
                box = element.bounding_box()
                if not box: return None
                
                x1 = max(0, box['x'] - pad_px)
                y1 = max(0, box['y'] - pad_px)
                x2 = min(width, box['x'] + box['width'] + pad_px)
                y2 = min(height, box['y'] + box['height'] + pad_px)
                
                new_w = x2 - x1
                new_h = y2 - y1
                center_x = x1 + (new_w / 2)
                center_y = y1 + (new_h / 2)
                
                return [class_id, center_x / width, center_y / height, new_w / width, new_h / height]

            def is_fully_visible(box, width, height, pad=4):
                return (box['x'] - pad >= 0 and 
                        box['y'] - pad >= 0 and 
                        (box['x'] + box['width'] + pad) <= width and 
                        (box['y'] + box['height'] + pad) <= height)
            
            label_data = []

            items = page.query_selector_all("div[trade-item-card]")
            for item in items:
                box = item.bounding_box()
                if box and is_fully_visible(box, width, height):
                    card_box = get_padded_yolo(item, 0, pad_px=4)
                    if card_box: label_data.append(card_box)

                    thumb = item.query_selector(".item-card-thumb-container") 
                    if thumb:
                        thumb_box = get_padded_yolo(thumb, 1, pad_px=4)
                        if thumb_box: label_data.append(thumb_box)
                    
                    name = item.query_selector(".item-card-name")
                    if name:
                        name_box = get_padded_yolo(name, 2, pad_px=4)
                        if name_box: label_data.append(name_box)
            
            robux_sections = page.query_selector_all(".robux-line:not(.total-value)")
            for section in robux_sections:
                box = section.bounding_box()
                if box and is_fully_visible(box, width, height, 8) and section.is_visible():
                    line_box = get_padded_yolo(section, 3, pad_px=8)
                    if line_box: label_data.append(line_box)

                    value_element = section.query_selector(".robux-line-value") 
                    if value_element:
                        value_box = get_padded_yolo(value_element, 4, pad_px=4)
                        if value_box: label_data.append(value_box)
            
            img_path = img_dir / f"{output_name}.png"
            page.screenshot(path=str(img_path))

            if random.random() < 0.60:
                img = cv2.imread(str(img_path))
                if img is not None:
                    if random.random() < 0.5:
                        quality = random.randint(60, 90) 
                        _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        img = cv2.imdecode(encimg, 1)
                    
                    if random.random() < 0.4:
                        alpha = random.uniform(0.8, 1.2)
                        beta = random.randint(-20, 20)
                        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                    
                    level = random.uniform(0.5, 2.5) 
                    noise = np.random.normal(0, level, img.shape).astype('float32')
                    img = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
                    cv2.imwrite(str(img_path), img)
            
            label_path = lbl_dir / f"{output_name}.txt"
            with open(label_path, "w") as f:
                for label in label_data:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

            browser.close()

    except Exception:
        print(f"Error generating task {task_id}:")
        traceback.print_exc()

def run_mass_generation(total_images=64, max_workers=4):
    db_path = TEMPLATES_DIR.parent.parent.parent / "assets" / "db.json"

    if not db_path.exists():
        db_path = Path(__file__).parent.parent.parent / "assets" / "db.json"

    with open(db_path, "r") as f:
        db = json.load(f)

    backgrounds_folder = Path("train/emulator/backgrounds")
    backgrounds_count = len([f for f in backgrounds_folder.iterdir() if f.is_file()]) - 1

    setup_dataset_directories(force_reset=True)

    print(f"Starting generation of {total_images} images using {max_workers} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_task, i, db, backgrounds_count) for i in range(total_images)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_images):
            pass

if __name__ == "__main__":
    run_mass_generation(total_images=64, max_workers=8)
