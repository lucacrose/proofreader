import os
import json
import torch
from PIL import Image
from ..core.config import DB_PATH, CACHE_PATH, ASSETS_PATH

class EmbeddingBuilder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_clip_embedding(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.numpy().flatten()

    def build(self):
        print(f"Starting build process...")
        print(f"Source Images: {ASSETS_PATH}")
        print(f"Item Database: {DB_PATH}")
        
        if not os.path.exists(DB_PATH):
            print(f"Error: Missing {DB_PATH}. Cannot map IDs to Names.")
            return
            
        with open(DB_PATH, "r") as f:
            items = json.load(f)
        
        embedding_bank = {}
        item_names = []
        
        if not os.path.exists(ASSETS_PATH):
            print(f"Error: Image directory {ASSETS_PATH} not found.")
            return

        image_files = [f for f in os.listdir(ASSETS_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(image_files)
        
        print(f"Found {total_files} images to process.")

        for i, filename in enumerate(image_files):
            item_id = os.path.splitext(filename)[0]

            item_info = next((item for item in items if str(item.get("id")) == item_id), None)
            
            if item_info:
                name = item_info["name"]
                img_path = os.path.join(ASSETS_PATH, filename)
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    embedding_bank[name] = self.get_clip_embedding(img)
                    item_names.append(name)
                    
                    if (i + 1) % 25 == 0 or (i + 1) == total_files:
                        print(f"Progress: {i + 1}/{total_files} items indexed...")
                except Exception as e:
                    print(f"Could not process {filename}: {e}")
            else:
                print(f"Warning: No database entry found for ID {item_id}")
        
        output_data = {
            'embeddings': embedding_bank, 
            'names': item_names
        }
        
        torch.save(output_data, CACHE_PATH)
        print(f"\n✅ Build Complete!")
        print(f"Target: {CACHE_PATH}")
        print(f"Total Embeddings Saved: {len(embedding_bank)}")
