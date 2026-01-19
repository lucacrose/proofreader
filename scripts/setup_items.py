import requests
import json
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from proofreader.core.config import THUMBNAILS_DIR, TRAIN_THUMBNAILS_DIR, DB_PATH, DEVICE

THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

def fetch_all_ids():
    unique_items = {}

    print("Fetching from Rolimons...")
    try:
        r_roli = requests.get("https://www.rolimons.com/itemapi/itemdetails")
        roli_data = r_roli.json()["items"]
        for item_id, details in roli_data.items():
            unique_items[int(item_id)] = details[0]
    except Exception as e:
        print(f"Rolimons fetch failed: {e}")
    
    print("Fetching from Roblox Catalog...")
    cursor = ""
    while True:
        url = f"https://catalog.roblox.com/v2/search/items/details?taxonomy=tZsUsd2BqGViQrJ9Vs3Wah&creatorName=Roblox&salesTypeFilter=2&includeNotForSale=true&limit=120&cursor={cursor}"
        resp = requests.get(url)
        if resp.status_code != 200: break
        
        data = resp.json()
        for item in data.get("data", []):
            item_id = int(item["id"])
            if item_id not in unique_items:
                unique_items[item_id] = item["name"]

        cursor = data.get("nextPageCursor")
        if not cursor: break
        print(f"Advanced to cursor: {cursor[:10]}...")

    return unique_items

def _download_single_image(thumb, pbar):
    target_id = thumb["targetId"]
    img_url = thumb["imageUrl"]
    img_path = THUMBNAILS_DIR / f"{target_id}.png"
    
    if not img_path.exists() and thumb.get("state") == "Completed":
        try:
            img_data = requests.get(img_url, timeout=10).content
            with open(img_path, "wb") as f:
                f.write(img_data)
        except Exception:
            pass
    
    pbar.update(1)

def download_thumbnails(item_ids):
    id_list = list(item_ids)
    to_download = [tid for tid in id_list if not (THUMBNAILS_DIR / f"{tid}.png").exists()]
    
    if not to_download:
        print("✅ All thumbnails already exist.")
        return
    
    all_thumbs = []
    print(f"📡 Fetching metadata for {len(to_download)} items...")

    for i in tqdm(range(0, len(to_download), 100), desc="Metadata"):
        batch = to_download[i : i + 100]
        ids_str = ",".join(map(str, batch))
        url = f"https://thumbnails.roblox.com/v1/assets?assetIds={ids_str}&size=420x420&format=Png&isCircular=false"
        
        try:
            resp = requests.get(url).json()
            all_thumbs.extend(resp.get("data", []))
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"💾 Downloading {len(all_thumbs)} images...")

    with tqdm(total=len(all_thumbs), desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=15) as executor:
            [executor.submit(_download_single_image, thumb, pbar) for thumb in all_thumbs]

    print("✨ Finished!")

def save_database(unique_items):
    out = [{"id": int(k), "name": v} for k, v in unique_items.items()]
    with open(DB_PATH, "w") as f:
        json.dump(out, f, separators=(',', ':'))
    print(f"Database saved to {DB_PATH}")

    return out

def organize_files(id_to_name):
    os.makedirs(TRAIN_THUMBNAILS_DIR, exist_ok=True)

    for filename in os.listdir(THUMBNAILS_DIR):
        if filename.endswith(".png"):
            item_id = filename.split(".")[0]
            if item_id in id_to_name:
                class_folder = os.path.join(TRAIN_THUMBNAILS_DIR, item_id)
                os.makedirs(class_folder, exist_ok=True)
                shutil.copy(os.path.join(THUMBNAILS_DIR, filename), class_folder)

    print(f"Organized images into {len(os.listdir(TRAIN_THUMBNAILS_DIR))} classes.")

if __name__ == "__main__":
    all_items = fetch_all_ids()

    database = save_database(all_items)

    organize_files(database)

    download_thumbnails(all_items.keys())
    
    print("Setup Complete! Your repo is now populated with data assets.")
