from playwright.sync_api import sync_playwright
from pathlib import Path
import json
import shutil
import random
import string
import concurrent.futures
from tqdm import tqdm
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

def setup_dataset_directories():
    dataset_path = BASE_DIR / "dataset"
    dirs = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "val" / "images",
        dataset_path / "val" / "labels",
    ]

    if dataset_path.exists():
        shutil.rmtree(dataset_path)
        
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    return dataset_path

DATASET_ROOT = setup_dataset_directories()
TEMPLATE_PATH = BASE_DIR / "trade_template.html"

# Pre-cache template files to avoid repeated disk I/O in threads
folder = Path(BASE_DIR / "templates")
TEMPLATE_FILES = [str(f.absolute()) for f in folder.iterdir() if f.is_file()]

def generate_random_username():
    length = random.randint(3, 20)
    chars = string.ascii_letters + string.digits + "_"
    return "".join(random.choice(chars) for _ in range(length))

def worker_task(task_id, db):
    split = "train" if random.random() < 0.8 else "val"
    output_name = f"trade_{task_id:05d}"
    
    # Prepare trade data
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
                    "img": f"../assets/images/{item['id']}.png"
                })

    # Initialize Playwright inside the thread for isolation
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        aspect_ratio = random.uniform(1.0, 2.4)
        
        # 2. Pick a random width within a realistic range for your GPU (RTX 5070)
        # We stay between 800 and 2560 to avoid tiny or massive browser windows.
        width = random.randint(800, 2560)
        
        # 3. Calculate height based on that aspect ratio
        height = int(width / aspect_ratio)
        
        # 4. Safety Check: If the height is too small/large, we cap it
        # Playwright/Browsers struggle with heights under 400 or over 1600
        height = max(400, min(height, 1600))
        context = browser.new_context(viewport={"width": width, "height": height})
        page = context.new_page()

        try:
            random_file = random.choice(TEMPLATE_FILES)
            page.goto(f"file://{random_file}")

            # Inject data and manipulate DOM
            page.evaluate("""
                ([newItems, userData, is_empty_trade]) => {
                    function getRandomInt(min, max) {
                        min = Math.ceil(min);
                        max = Math.floor(max);
                        return Math.floor(Math.random() * (max - min + 1)) + min;
                    }
                        
                    function getRandomNumberEqualDigits(minDigits = 1, maxDigits = 5) {
                        const digits = getRandomInt(minDigits, maxDigits);
                        const min = digits === 1 ? 0 : Math.pow(10, digits - 1);
                        const max = Math.pow(10, digits) - 1;
                        return getRandomInt(min, max);
                    }
                          
                    // Function to generate a random RGBA color
                    function getRandomColor(alphaMin = 0.3, alphaMax = 1) {
                        const r = Math.floor(Math.random() * 256); // 0-255
                        const g = Math.floor(Math.random() * 256);
                        const b = Math.floor(Math.random() * 256);
                        const a = (Math.random() * (alphaMax - alphaMin) + alphaMin).toFixed(2); // opacity
                        return `rgba(${r}, ${g}, ${b}, ${a})`;
                    }

                    // Total number of backgrounds you have
                    const totalBackgrounds = 99;

                    // Pick a random number between 1 and totalBackgrounds
                    const randomIndex = Math.floor(Math.random() * totalBackgrounds) + 1;

                    // Pad with leading zeros if needed (e.g., 001, 002, ...)
                    const paddedIndex = String(randomIndex).padStart(3, '0');

                    // Build the URL
                    const randomBgUrl = `url('../assets/backgrounds/unsplash_${paddedIndex}.jpg')`;

                    // Apply to container-main
                    if (Math.random() < 0.75) {
                        const container = document.querySelector(".container-main");
                        container.style.backgroundImage = randomBgUrl;
                        container.style.backgroundSize = "cover";
                        container.style.backgroundPosition = "center";
                        container.style.backgroundRepeat = "no-repeat";
                    }

                    // Set fully random color with opacity for .content
                    if (Math.random() < 0.75) {
                        const content = document.querySelector(".content");
                        content.style.backgroundColor = getRandomColor();
                        content.style.color = getRandomColor(0.95, 1);
                    }
                    
                    const nameElements = document.querySelectorAll('.paired-name .element');
                    if(nameElements.length >= 2) {
                        nameElements[0].innerText = userData.display;
                        nameElements[1].innerText = userData.handle;
                    }
                          
                    function randomAlphanumericWithSpaces(min = 2, max = 48) {
                        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
                        const length = Math.floor(Math.random() * (max - min + 1)) + min;

                        let result = chars[Math.floor(Math.random() * chars.length)];

                        for (let i = 1; i < length; i++) {
                            if (result[result.length - 1] === " ") {
                            result += chars[Math.floor(Math.random() * chars.length)];
                            } else {
                            result += Math.random() < 0.15
                                ? " "
                                : chars[Math.floor(Math.random() * chars.length)];
                            }
                        }

                        return result;
                    }
                          
                    const cards = document.querySelectorAll('div[trade-item-card]');
                    cards.forEach((card, index) => {
                        const isFirstSide = index < 4;
                        const sideIndex = isFirstSide ? 0 : 1;
                        const itemIndex = isFirstSide ? index : index - 4;
                        const data = newItems[sideIndex][itemIndex];
                        const hideName = Math.random() < 0.2;
                        const hideThumb = Math.random() < 0.2;

                        if (data) {
                            card.style.visibility = "visible";
                            card.style.opacity = "1";
                            card.setAttribute("data-item-id", data.id);

                            const img = card.querySelector('img');
                            if (img) img.src = data.img;

                            const priceLabel = card.querySelector('.text-robux');
                            if (priceLabel) {
                                priceLabel.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();

                                // 🔁 35% chance to clone price line
                                const CHANCE = 0.35;
                                const priceLine = priceLabel.closest('.item-card-price');

                                if (
                                    priceLine &&
                                    Math.random() < CHANCE &&
                                    !priceLine.nextElementSibling?.classList.contains('item-card-price')
                                ) {
                                    const clone = priceLine.cloneNode(true);

                                    // optional: tweak cloned value slightly
                                    const cloneValue = clone.querySelector('.text-robux');
                                    if (cloneValue) {
                                        const base = Number(priceLabel.innerText.replace(/,/g, ''));
                                        cloneValue.innerText = getRandomNumberEqualDigits(1, 9).toLocaleString();
                                    }

                                    priceLine.parentElement.insertBefore(clone, priceLine.nextSibling);
                                }
                            }

                            const nameLabel = card.querySelector('.item-card-name');
                            if (nameLabel) {
                                nameLabel.innerText = randomAlphanumericWithSpaces();
                                nameLabel.style.lineHeight = `${Math.floor(Math.random() * 17) + 12}px`; // 12–28px
                            }

                            if (hideName && nameLabel) nameLabel.style.display = "none";
                            if (hideThumb && img) img.parentElement.parentElement.parentElement.parentElement.style.display = "none";
                        } else {
                            card.style.opacity = "0"; 
                            card.setAttribute("data-item-id", "");
                        }
                    });
                    
                    const robuxLines = document.querySelectorAll(".robux-line");
                    if(robuxLines.length >= 3) {
                        robuxLines[0].style.display = is_empty_trade ? "none" : (Math.random() < 0.25 ? "none" : "");
                        robuxLines[2].style.display = is_empty_trade ? "none" : (Math.random() < 0.25 ? "none" : "");
                    }
                          
                    document.querySelectorAll(".robux-line-value").forEach(el => {
                        el.textContent = getRandomNumberEqualDigits(1, 10).toLocaleString();
                    });

                    document.querySelectorAll(".limited-icon-container").forEach(container => {
                        const numberContainer = container.querySelector(".limited-number-container");
                        const numberSpan = container.querySelector(".limited-number");
                        if (!numberContainer || !numberSpan) return;
                        const show = Math.random() < 0.5;
                        if (show) {
                            numberContainer.style.display = "";
                            numberSpan.style.display = "";
                            numberSpan.textContent = getRandomNumberEqualDigits(1, 7);
                        } else {
                            numberContainer.style.display = "none";
                            numberSpan.style.display = "none";
                        }
                    });
                          
                    document.querySelectorAll('.item-card-name, .item-card-price').forEach(el => {
                        const offsetLeft = Math.floor(Math.random() * 25); // 0–24 px
                        const offsetTop = Math.floor(Math.random() * 13); // 0–12 px
                        el.style.marginLeft = `${offsetLeft}px`;
                        el.style.marginTop = `${offsetTop}px`;
                    });
                    
                    const withColon = text =>
                    Math.random() < 0.5 ? `${text}:` : text;

                    document.querySelectorAll('.trade-list-detail-offer').forEach(offer => {
                        if (Math.random() > 0.3) return;

                        if (offer.querySelector('.robux-line.total-rap')) return;

                        const robuxLines = [...offer.querySelectorAll('.robux-line')];

                        const totalValueLine = robuxLines.find(line =>
                            line.querySelector('.text-lead')?.textContent
                            .replace(/:/g, '')
                            .trim() === 'Total Value'
                        );

                        if (!totalValueLine) return;

                        // Randomize Total Value label colon
                        const valueLabel = totalValueLine.querySelector('.text-lead');
                        if (valueLabel) {
                            valueLabel.textContent = withColon('Total Value');
                        }

                        // Calculate RAP
                        const rapValue = [...offer.querySelectorAll('.item-card-price .text-robux')]
                            .reduce((sum, el) => sum + Number(el.textContent.replace(/,/g, '')), 0)
                            .toLocaleString();

                        // Create RAP line
                        const rapLine = document.createElement('div');
                        rapLine.className = 'robux-line total-rap';
                        rapLine.innerHTML = `
                            <span class="text-lead">${withColon('Total RAP')}</span>
                            <span class="robux-line-amount">
                            <span class="icon-robux-16x16"></span>
                            <span class="text-robux-lg robux-line-value">${rapValue}</span>
                            </span>
                        `;

                        totalValueLine.parentElement.insertBefore(rapLine, totalValueLine);
                    });
                }
            """, [trade_input, {"display": generate_random_username(), "handle": generate_random_username()}, is_empty_trade])

            def get_padded_yolo(element, class_id, pad_px=2):
                box = element.bounding_box()
                if not box: return None
                
                # Add padding in pixels
                x1 = max(0, box['x'] - pad_px)
                y1 = max(0, box['y'] - pad_px)
                x2 = min(width, box['x'] + box['width'] + pad_px)
                y2 = min(height, box['y'] + box['height'] + pad_px)
                
                # Calculate new width/height and center
                new_w = x2 - x1
                new_h = y2 - y1
                center_x = x1 + (new_w / 2)
                center_y = y1 + (new_h / 2)
                
                return [
                    class_id,
                    center_x / width,
                    center_y / height,
                    new_w / width,
                    new_h / height
                ]
            
            # Updated visibility check that accounts for the padding you intend to add
            def is_fully_visible(box, width, height, pad=4):
                return (box['x'] - pad >= 0 and 
                        box['y'] - pad >= 0 and 
                        (box['x'] + box['width'] + pad) <= width and 
                        (box['y'] + box['height'] + pad) <= height)
            # Extract labels
            label_data = []
            # Process Item Cards (Class 0)
            items = page.query_selector_all("div[trade-item-card]")
            for item in items:
                box = item.bounding_box()
                if box and is_fully_visible(box, width, height) and item.get_attribute("data-item-id"):
                    yolo_box = get_padded_yolo(item, 0, pad_px=4) # Cards don't need much pad
                    if yolo_box: label_data.append(yolo_box)

                    thumb = item.query_selector(".item-card-thumb-container") 
                    if thumb:
                        thumb_box = get_padded_yolo(thumb, 1, pad_px=4)
                        if thumb_box: label_data.append(thumb_box)

                    # 3. Sub-Object: Name Label (Class 2)
                    # Target the specific text element for the name
                    name = item.query_selector(".item-card-name")
                    if name:
                        name_box = get_padded_yolo(name, 2, pad_px=4)
                        if name_box: label_data.append(name_box)
            
            # Process Robux Lines (Class 1)
            # Notice the selection excludes the last child to avoid "Total Value"
            robux_sections = page.query_selector_all(".robux-line:first-child")
            for section in robux_sections:
                box = section.bounding_box()
                if box and is_fully_visible(box, width, height, 8) and section.is_visible():
                    # 1. Parent Robux Line (Class 3)
                    line_box = get_padded_yolo(section, 3, pad_px=8)
                    if line_box: label_data.append(line_box)

                    # 2. Sub-Object: Robux Value (Class 4)
                    # Target ONLY the span/div that contains the number string
                    value_element = section.query_selector(".robux-line-value") 
                    if value_element:
                        # Low padding here so the OCR doesn't see the "R$" icon
                        value_box = get_padded_yolo(value_element, 4, pad_px=4)
                        if value_box: label_data.append(value_box)
            
            # Save results
            img_path = DATASET_ROOT / split / "images" / f"{output_name}.png"
            page.screenshot(path=str(img_path))

            if random.random() < 0.60: # Reduced frequency slightly
                img = cv2.imread(str(img_path))
                
                # --- A. Realistic JPEG Compression (Calmed) ---
                # Quality 60-90 is the "sweet spot" for web artifacts without destroying text
                if random.random() < 0.5:
                    quality = random.randint(60, 90) 
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    _, encimg = cv2.imencode('.jpg', img, encode_param)
                    img = cv2.imdecode(encimg, 1)

                # --- B. Contrast & Brightness (Normalized) ---
                # Avoids "pitch black" or "pure white" results
                if random.random() < 0.4:
                    alpha = random.uniform(0.8, 1.2) # Contrast
                    beta = random.randint(-20, 20)   # Brightness
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

                # --- C. Subtle Rescale Blur ---
                # Min scale 0.75 ensures text doesn't turn into a single pixel row
                if random.random() < 0.4:
                    scale = random.uniform(0.75, 0.95) 
                    h, w = img.shape[:2]
                    # Using INTER_AREA for downscaling is cleaner than NEAREST
                    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

                # --- D. Natural Grain ---
                # Level 0.5 to 2.5 is visible but doesn't create "ghost" text
                level = random.uniform(0.5, 2.5) 
                noise = np.random.normal(0, level, img.shape).astype('float32')
                img = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')

                # --- E. Selective Sharpening ---
                # Removed Median Blur (5) as it was the main cause of unreadability
                if random.random() < 0.3:
                    # Light sharpening helps the model define edges at 640px
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    img = cv2.filter2D(img, -1, kernel)

                # --- F. Subtle Color Jitter ---
                if random.random() < 0.3:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.uniform(-10, 10), 0, 255)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-10, 10), 0, 255)
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                cv2.imwrite(str(img_path), img)

            label_path = DATASET_ROOT / split / "labels" / f"{output_name}.txt"
            with open(label_path, "w") as f:
                for label in label_data:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        except Exception as e:
            print(f"Error generating {output_name}: {e}")
        finally:
            browser.close()

def run_mass_generation(total_images=1000, max_workers=4):
    # Load database once
    db_path = BASE_DIR / "assets" / "db.json"
    with open(db_path, "r") as f:
        db = json.load(f)

    print(f"Starting generation of {total_images} images using {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to executor
        futures = [executor.submit(worker_task, i, db) for i in range(total_images)]
        
        # Use tqdm for a nice progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), total=total_images):
            pass

if __name__ == "__main__":
    # Adjust max_workers based on your CPU/RAM. 
    # Browser instances are heavy; 4-6 is usually a safe range.
    run_mass_generation(total_images=8192, max_workers=24)