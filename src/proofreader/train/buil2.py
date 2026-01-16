import torch
import json

# 1. Load your mapping to keep names synced
with open("class_mapping.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 2. Load the trained model weights
checkpoint = torch.load("item_clip_best.pt")

# 3. Extract the 'item_prototypes' (these ARE your embeddings)
# Note: Ensure the key matches your model's attribute name (usually 'item_prototypes.weight')
embeddings_matrix = checkpoint['item_prototypes.weight']

# 4. Normalize them (so Cosine Similarity is just a dot product)
embeddings_normalized = torch.nn.functional.normalize(embeddings_matrix, p=2, dim=1)

# 5. Save as your final bank
bank = {
    "embeddings": embeddings_normalized.cpu(),
    "mapping": idx_to_class
}

torch.save(bank, "item_embeddings_bank.pt")
print(f"Successfully exported {len(idx_to_class)} embeddings to bank.")