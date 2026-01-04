import os
import json
import torch
from PIL import Image
from ..core.config import DB_PATH, CACHE_PATH, THUMBNAILS_DIR, BUILDER_BATCH_SIZE

class EmbeddingBuilder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def get_clip_embedding(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()

    def build(self, batch_size=BUILDER_BATCH_SIZE):
        self.model.eval()
        print(f"Starting build process...")
        print(f"Source Images: {THUMBNAILS_DIR}")
        print(f"Item Database: {DB_PATH}")
        
        if not os.path.exists(DB_PATH):
            print(f"Error: Missing {DB_PATH}. Cannot map IDs to Names.")
            return
            
        with open(DB_PATH, "r") as f:
            items = json.load(f)
        
        embedding_bank = {}
        item_names = []
        
        if not os.path.exists(THUMBNAILS_DIR):
            print(f"Error: Image directory {THUMBNAILS_DIR} not found.")
            return

        image_files = [f for f in os.listdir(THUMBNAILS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(image_files)
        
        embedding_bank = {}
        item_names = []

        for i in range(0, total_files, batch_size):
            batch_files = image_files[i : i + batch_size]
            batch_imgs = []
            batch_item_names = []

            for filename in batch_files:
                item_id = os.path.splitext(filename)[0]
                item_info = next((item for item in items if str(item.get("id")) == item_id), None)
                
                if item_info:
                    try:
                        img_path = os.path.join(THUMBNAILS_DIR, filename)
                        raw_img = Image.open(img_path)

                        if raw_img.mode in ("RGBA", "P"):
                            bg = Image.new("RGB", raw_img.size, (255, 255, 255))
                            bg.paste(raw_img.convert("RGBA"), (0, 0), raw_img.convert("RGBA"))
                            img = bg
                        else:
                            img = raw_img.convert("RGB")
                            
                        batch_imgs.append(img)
                        batch_item_names.append(item_info["name"])
                    except Exception as e:
                        print(f"Could not load {filename}: {e}")

            if not batch_imgs:
                continue
            try:
                inputs = self.processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.model.device)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                
                features_numpy = features.cpu().numpy()
                for name, emb in zip(batch_item_names, features_numpy):
                    embedding_bank[name] = emb
                    item_names.append(name)
                
                print(f"Progress: {min(i + batch_size, total_files)}/{total_files} items indexed...")
            except Exception as e:
                print(f"Batch processing error: {e}")
        
        output_data = {
            'embeddings': embedding_bank, 
            'names': item_names
        }
        
        torch.save(output_data, CACHE_PATH)
        print(f"\n✅ Build Complete!")
        print(f"Target: {CACHE_PATH}")
        print(f"Total Embeddings Saved: {len(embedding_bank)}")
