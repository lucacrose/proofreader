import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import random
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from proofreader.core.config import DATASET_ROOT, DEVICE

# Assuming these are imported from your project structure
# from proofreader.core.config import DATASET_ROOT, DEVICE

# --- CONFIGURATION (Adjust if not using config file) ---
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

class ResolutionGeneralizer:
    """Simulates low-quality screenshots by downscaling and upscaling."""
    def __init__(self, min_res=48, max_res=160):
        self.min_res = min_res
        self.max_res = max_res

    def __call__(self, img):
        if random.random() < 0.7:
            current_w, current_h = img.size
            res = random.randint(self.min_res, self.max_res)
            interp = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            img = img.resize((res, res), resample=interp)
            img = img.resize((current_w, current_h), resample=interp)
        return img

def get_sampler(subset):
    """
    Creates a sampler for 2400 classes.
    Handles 'Subset' objects by mapping back to original dataset targets.
    """
    # Pull targets from the underlying dataset using the subset's indices
    targets = np.array([subset.dataset.targets[i] for i in subset.indices])
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    
    # Calculate weights per class (1/count)
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[targets])
    
    return WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

def train_model():
    # 1. Advanced Augmentation
    data_transforms = transforms.Compose([
        ResolutionGeneralizer(min_res=48, max_res=160),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])

    # 2. Dataset Setup
    dataset_path = Path(DATASET_ROOT) / "classification"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    full_dataset = datasets.ImageFolder(root=str(dataset_path), transform=data_transforms)
    num_classes = len(full_dataset.classes)
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create Balanced Sampler for Training
    train_sampler = get_sampler(train_dataset)

    # Multi-worker DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=8, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 3. Model Initialization (Swin-T)
    print(f"Loading Swin-T for {num_classes} classes...")
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(DEVICE)

    # 4. Optimization Suite
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler('cuda')

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # AMP Mixed Precision
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"End of Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {accuracy:.2f}%")

    # 6. Save Artifacts
    weights_dir = Path("src/assets/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), weights_dir / "item_classifier.pt")
    
    class_map = {v: k for k, v in full_dataset.class_to_idx.items()}
    with open("src/assets/class_map.json", "w") as f:
        json.dump(class_map, f)

    print("Success: Weights and class map saved.")

# Critical for Windows Multiprocessing
if __name__ == "__main__":
    # Provides support for frozen executables and general process stability
    torch.multiprocessing.freeze_support()
    train_model()