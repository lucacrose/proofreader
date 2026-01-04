import requests
import json
import os
from transformers import CLIPProcessor, CLIPModel
from proofreader.core.config import THUMBNAILS_DIR, DB_PATH, CACHE_PATH, DEVICE
from proofreader.train.builder import EmbeddingBuilder

embedding_builder = EmbeddingBuilder(CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True))

THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

def fetch_all_ids():
    """Combines IDs from Rolimons and Roblox Catalog, removing overlaps."""
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

def download_thumbnails(item_ids):
    id_list = list(item_ids)
    print(f"Total unique items to process: {len(id_list)}")

    for i in range(0, len(id_list), 30):
        batch = id_list[i : i + 30]
        ids_str = ",".join(map(str, batch))

        thumb_url = f"https://thumbnails.roblox.com/v1/assets?assetIds={ids_str}&size=250x250&format=Png&isCircular=false"
        try:
            resp = requests.get(thumb_url).json()
            for thumb in resp.get("data", []):
                target_id = thumb["targetId"]
                img_url = thumb["imageUrl"]

                img_path = THUMBNAILS_DIR / f"{target_id}.png"
                if not img_path.exists() and thumb["state"] == "Completed":
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
            
            print(f"Processed batch {i//30 + 1}/{(len(id_list)//30)+1}")
        except Exception as e:
            print(f"Error in batch: {e}")

def save_database(unique_items):
    out = [{"id": k, "name": v} for k, v in unique_items.items()]
    with open(DB_PATH, "w") as f:
        json.dump(out, f, separators=(',', ':'))
    print(f"Database saved to {DB_PATH}")

if __name__ == "__main__":
    all_items = fetch_all_ids()

    save_database(all_items)

    download_thumbnails(all_items.keys())

    if not os.path.exists(CACHE_PATH):
            embedding_builder.build()
    elif os.path.getmtime(THUMBNAILS_DIR) > os.path.getmtime(CACHE_PATH):
        embedding_builder.build()
    
    print("Setup Complete! Your repo is now populated with data assets.")
