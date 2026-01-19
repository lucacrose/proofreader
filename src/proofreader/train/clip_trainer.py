import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from proofreader.core.config import CLASS_MAP_PATH, CLIP_BEST_PATH, DATASET_ROOT
import os
import json

MODEL_ID = "openai/clip-vit-base-patch32"
EPOCHS = 20
BATCH_SIZE = 48 
LEARNING_RATE = 1e-5
EMBEDDING_DIM = 512
WEIGHT_DECAY = 0.1
PATIENCE = 3        # Stop if no improvement for 3 epochs
MIN_DELTA = 0.1     # Minimum % improvement to be considered "better"

class CLIPItemEmbedder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID)
        self.item_prototypes = nn.Embedding(num_classes, EMBEDDING_DIM)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, pixel_values, item_ids):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = outputs.image_embeds 
        
        label_embeds = self.item_prototypes(item_ids)
        label_embeds = F.normalize(label_embeds, p=2, dim=-1)
        
        return image_embeds, label_embeds, self.logit_scale.exp()

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = 0
        self.best_state = None

    def check(self, current_accuracy, model):
        if current_accuracy > (self.best_accuracy + self.min_delta):
            self.best_accuracy = current_accuracy
            # Handle potential torch.compile wrapper (though disabled for Windows)
            self.best_state = getattr(model, "_orig_mod", model).state_dict()
            self.counter = 0
            return False, True # Don't stop, but is a new best
        else:
            self.counter += 1
            return (self.counter >= self.patience), False

def get_transforms():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # Resolution Crush
        transforms.RandomApply([
            transforms.RandomChoice([transforms.Resize(128), transforms.Resize(64)]),
            transforms.Resize(224),
        ], p=0.3),
        # Gaussian Blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
        ], p=0.2),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

def train_clip():
    torch.backends.cudnn.benchmark = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = f"{DATASET_ROOT}/classification"
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=get_transforms())
    num_classes = len(full_dataset.classes)

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(full_dataset.class_to_idx, f, separators=(",", ":"))

    train_size = int(0.95 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=os.cpu_count(), pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    model = CLIPItemEmbedder(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    stopper = EarlyStopper(patience=PATIENCE, min_delta=MIN_DELTA)

    print(f"Starting training for {num_classes} classes...")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                img_emb, lbl_emb, scale = model(images, labels)

                logits = scale * img_emb @ lbl_emb.t()
                ground_truth = torch.arange(len(images), device=device)
                loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad(), autocast('cuda'):
            all_ids = torch.arange(num_classes).to(device)
            prototypes = F.normalize(model.item_prototypes(all_ids), p=2, dim=-1)
            
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                img_emb, _, _ = model(images, labels)
                preds = (img_emb @ prototypes.t()).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        stop_now, is_best = stopper.check(val_acc, model)
        if is_best:
            torch.save(stopper.best_state, CLIP_BEST_PATH)
            print("Successfully saved new best model weights.")
        
        if stop_now:
            print(f"Stopping early. Best Accuracy was {stopper.best_accuracy:.2f}%")
            break

    print("Training finished.")

if __name__ == "__main__":
    train_clip()
