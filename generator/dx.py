import requests
import json
import os
import time

# Configuration
DB_PATH = "generator/assets/db.json"
IMAGE_DIR = "generator/assets/images"  # Current directory where images are saved
BATCH_SIZE = 120 # Roblox API limit for batch requests

def load_db():
    if not os.path.exists(DB_PATH):
        print(f"Warning: {DB_PATH} not found. Starting with empty list.")
        return []
    
    with open(DB_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON. Starting with empty list.")
            return []

def get_downloaded_ids():
    ids = set()
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".png"):
            # Extract ID from filename "12345.png" -> "12345"
            try:
                item_id = filename.split(".")[0]
                # Ensure it's a number before adding
                if item_id.isdigit():
                    ids.add(str(item_id))
            except:
                continue
    return ids

def fetch_names(missing_ids):
    url = "https://catalog.roblox.com/v1/catalog/items/details"
    new_items = []
    
    # Convert set to list for indexing
    missing_list = list(missing_ids)
    total = len(missing_list)
    
    print(f"Fetching details for {total} items...")

    # Session helps maintain cookies if needed, though mostly for the header update
    session = requests.Session()
    
    for i in range(0, total, BATCH_SIZE):
        batch = missing_list[i : i + BATCH_SIZE]
        
        # Construct payload for multi-get
        payload = {
            "items": [{"itemType": "Asset", "id": int(uid)} for uid in batch]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            response = session.post(url, json=payload, headers=headers)
            
            # --- CSRF HANDLING START ---
            if response.status_code == 403 and "x-csrf-token" in response.headers:
                print("CSRF Token challenge received. Retrying with new token...")
                # Update headers with the new token provided by Roblox
                headers["x-csrf-token"] = response.headers["x-csrf-token"]
                # Retry the request
                response = session.post(url, json=payload, headers=headers)
            # --- CSRF HANDLING END ---

            if response.status_code == 429:
                print("Rate limit hit. Waiting 5 seconds...")
                time.sleep(5)
                # Retry once for rate limit if you want, or just skip
                continue
                
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                continue

            data = response.json()
            
            if "data" in data:
                for item in data["data"]:
                    new_items.append({
                        "id": str(item["id"]),
                        "name": item.get("name", "Unknown Name") # Safety fallback
                    })
                    print(f"Found: {item.get('name')} ({item['id']})")
            
            # Sleep briefly to be polite to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Exception during request: {e}")

    return new_items

def main():
    # 1. Load existing DB
    current_db = load_db()
    
    # Create a set of existing IDs for fast lookup
    # Rolimons IDs in your script seem to be keys, but let's ensure we handle strings
    existing_ids = set(str(item["id"]) for item in current_db)
    
    # 2. Get IDs from files
    downloaded_ids = get_downloaded_ids()
    
    # 3. Find missing
    missing_ids = downloaded_ids - existing_ids
    
    if not missing_ids:
        print("No missing names found! All downloaded images are in db.json.")
        return

    print(f"Found {len(missing_ids)} items with images but no names.")
    
    # 4. Fetch names from Roblox
    new_entries = fetch_names(missing_ids)
    
    # 5. Update and Save
    if new_entries:
        current_db.extend(new_entries)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        with open(DB_PATH, "w") as f:
            json.dump(current_db, f, separators=(',', ':'))
            
        print(f"Successfully added {len(new_entries)} items to {DB_PATH}")
    else:
        print("No new items were successfully fetched.")

if __name__ == "__main__":
    main()