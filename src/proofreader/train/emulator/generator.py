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

def clean_and_save_labels(page, width, height):
    label_data = []

    chat_bar = page.query_selector("#chat-main")
    chat_box = chat_bar.bounding_box() if chat_bar and chat_bar.is_visible() else None

    def is_overlapped_by_chat(box, class_id):
        if class_id not in [2, 4] or not box or not chat_box:
            return False
        
        return not (
            box['x'] + box['width'] < chat_box['x'] or
            box['x'] > chat_box['x'] + chat_box['width'] or
            box['y'] + box['height'] < chat_box['y'] or
            box['y'] > chat_box['y'] + chat_box['height']
        )

    def get_valid_yolo_data(box, class_id, pad=0):
        if not box:
            return None

        x1_raw = box['x'] - pad
        y1_raw = box['y'] - pad
        x2_raw = box['x'] + box['width'] + pad
        y2_raw = box['y'] + box['height'] + pad

        if class_id in [1, 3]:
            if x1_raw < 0 or y1_raw < 0 or x2_raw > width or y2_raw > height:
                return None
            
        if is_overlapped_by_chat(box, class_id):
            return None

        x1 = max(0, x1_raw)
        y1 = max(0, y1_raw)
        x2 = min(width, x2_raw)
        y2 = min(height, y2_raw)

        actual_h = y2 - y1
        if class_id in [5, 6]:
            if actual_h < (box['height'] * 0.7):
                return None

        nw, nh = x2 - x1, y2 - y1
        if nw <= 1 or nh <= 1:
            return None

        return [
            class_id,
            (x1 + nw/2) / width,
            (y1 + nh/2) / height,
            nw / width,
            nh / height
        ], (x1, y1, x2, y2)
    
    items = page.query_selector_all("div[trade-item-card]")
    for item in items:
        thumb = item.query_selector(".item-card-thumb-container")
        name = item.query_selector(".item-card-name")
        
        t_box = thumb.bounding_box() if thumb and thumb.is_visible() else None
        n_box = name.bounding_box() if name and name.is_visible() else None

        t_data = get_valid_yolo_data(t_box, 1, pad=4)
        n_data = get_valid_yolo_data(n_box, 2, pad=4)

        if t_data:
            u_x1, u_y1, u_x2, u_y2 = t_data[1]
            if n_data:
                u_x1 = min(u_x1, n_data[1][0])
                u_x2 = max(u_x2, n_data[1][2])
                u_y2 = n_data[1][3]

            nw, nh = u_x2 - u_x1, u_y2 - u_y1
            label_data.append([0, (u_x1 + nw/2)/width, (u_y1 + nh/2)/height, nw/width, nh/height])
            label_data.append(t_data[0])
            if n_data: label_data.append(n_data[0])

    for section in page.query_selector_all(".robux-line"):
        if section.is_visible() and "Robux Offered" in section.inner_text():
            res = get_valid_yolo_data(section.bounding_box(), 3, pad=6)
            if res: 
                label_data.append(res[0])
                val_el = section.query_selector(".robux-line-value")
                if val_el and val_el.is_visible():
                    res_v = get_valid_yolo_data(val_el.bounding_box(), 4, pad=4)
                    if res_v: label_data.append(res_v[0])

    for header in page.query_selector_all("h3.trade-list-detail-offer-header"):
        if not header.is_visible(): continue
        text = header.inner_text().lower()
        cid = 5 if "gave" in text else 6 if "received" in text else None
        if cid:
            res_h = get_valid_yolo_data(header.bounding_box(), cid, pad=4)
            if res_h: label_data.append(res_h[0])

    return label_data

def process_batch(batch_ids, db, backgrounds_count, progress_counter):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
            )
            
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
    split = "train" if random.random() < GENERATOR_CONFIG["train_split_fraction"] else "val"
    output_name = f"trade_{task_id:05d}"
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    trade_input = [[], []]
    is_empty_trade = random.random() < GENERATOR_CONFIG["empty_trade_chance"]
    if not is_empty_trade:
        for side in [0, 1]:
            num_items = random.randint(0, 4)
            for _ in range(num_items):
                item = random.choice(db)
                trade_input[side].append(f"../../../../assets/thumbnails/{item['id']}.png")
    
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

    page.evaluate("""
        async () => {
            const imgs = Array.from(document.querySelectorAll('img'));
            const promises = imgs.map(img => {
                if (img.complete) return Promise.resolve();
                return new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = resolve; // Continue even if image fails
                });
            });
            await Promise.all(promises);
            
            // Final safety: Wait for the browser to paint
            await new Promise(r => requestAnimationFrame(r));
        }
    """)

    label_data = clean_and_save_labels(page, width, height)
    
    img_buffer = page.screenshot(type="jpeg", quality=100)
    nparr = np.frombuffer(img_buffer, np.uint8)
    full_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
                pad = 4
                max_offset = 5
                off_x = random.randint(-max_offset, max_offset)
                off_y = random.randint(-max_offset, max_offset)

                x1, y1 = int(box['x'] - pad + off_x), int(box['y'] - pad + off_y)
                x2, y2 = int(box['x'] + box['width'] + pad + off_x), int(box['y'] + box['height'] + pad + off_y)
                if 0 <= x1 and 0 <= y1 and x2 <= width and y2 <= height:
                    crop = full_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        class_dir = DATASET_ROOT / "classification" / item_id
                        class_dir.mkdir(parents=True, exist_ok=True)
                        if random.random() < 0.3:
                            brightness = random.uniform(0.7, 1.3)
                            crop = cv2.convertScaleAbs(crop, alpha=brightness, beta=0)
                        
                        if random.random() < 0.2:
                            k_size = random.choice([3, 5])
                            crop = cv2.GaussianBlur(crop, (k_size, k_size), 0)
                        
                        q = random.randint(70, 95)
                        cv2.imwrite(str(class_dir / f"{output_name}_{i}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    
    if random.random() < 0.60:
        if random.random() < 0.5:
            q = random.randint(60, 90)
            _, enc = cv2.imencode('.jpg', full_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            full_img = cv2.imdecode(enc, 1)

        if random.random() < 0.4:
            full_img = cv2.convertScaleAbs(full_img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        
        noise = np.random.normal(0, random.uniform(0.5, 2.5), full_img.shape).astype('float32')
        full_img = np.clip(full_img.astype('float32') + noise, 0, 255).astype('uint8')
    
    cv2.imwrite(str(img_dir / f"{output_name}.jpg"), full_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    with open(lbl_dir / f"{output_name}.txt", "w") as f:
        for label in label_data:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def run_mass_generation(total_images=16384, max_workers=24):
    with open(DB_PATH, "r") as f:
        db = json.load(f)
    
    setup_dataset_directories(force_reset=True)

    batch_size = 500
    all_ids = list(range(total_images))
    chunks = [all_ids[i:i + batch_size] for i in range(0, len(all_ids), batch_size)]

    backgrounds_count = len([f for f in BACKGROUNDS_DIR.iterdir() if f.is_file()])

    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0)

    print(f"Generating {total_images} images using {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_batch, chunk, db, backgrounds_count, progress_counter) 
            for chunk in chunks
        ]

        with tqdm(total=total_images, desc="Generating Images") as pbar:
            last_val = 0
            while True:
                done, not_done = concurrent.futures.wait(futures, timeout=0.5)

                current_val = progress_counter.value
                pbar.update(current_val - last_val)
                last_val = current_val
                
                if len(not_done) == 0:
                    break

if __name__ == "__main__":
    run_mass_generation()
