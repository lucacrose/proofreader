import torch
import shutil
from pathlib import Path

# --- BASE PATHS ---
# Resolves to the 'proofreader' root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# --- ASSETS & MODELS ---
ASSETS_PATH = BASE_DIR / "assets"
MODEL_PATH = ASSETS_PATH / "weights" / "yolo_v1.pt"
DB_PATH = ASSETS_PATH / "db.json"
CACHE_PATH = ASSETS_PATH / "embedding_bank.pt"
THUMBNAILS_DIR = ASSETS_PATH / "thumbnails"

# --- TRAINING & EMULATOR ---
TRAIN_DIR = BASE_DIR / "train"
DATA_YAML_PATH = TRAIN_DIR / "config" / "data.yaml"
DATASET_ROOT = TRAIN_DIR / "dataset"

EMULATOR_DIR = TRAIN_DIR / "emulator"
TEMPLATES_DIR = EMULATOR_DIR / "templates"
AUGMENTER_PATH = EMULATOR_DIR / "augmenter.js"
DEFAULT_TEMPLATE = TEMPLATES_DIR / "trade_ui.html"

# --- HYPERPARAMETERS (Training Settings) ---
TRAINING_CONFIG = {
    "epochs": 100,             # Number of times the model sees the whole dataset
    "batch_size": 16,          # Number of images processed at once
    "img_size": 640,           # Standard YOLO resolution
    "patience": 10,            # Stop early if no improvement for 10 epochs
    "close_mosaic_epochs": 10  # Disable mosaic augmentation for the last N epochs
}

# --- HARDWARE SETTINGS ---
# Automatically detects if a GPU is available for faster training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- DYNAMIC ASSETS ---
# Resolve template files once during import
if TEMPLATES_DIR.exists():
    TEMPLATE_FILES = [
        str(f.resolve())
        for f in TEMPLATES_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
else:
    TEMPLATE_FILES = []

# --- UTILITIES ---

def setup_dataset_directories(force_reset=False):
    dirs = [
        DATASET_ROOT / "train" / "images",
        DATASET_ROOT / "train" / "labels",
        DATASET_ROOT / "val" / "images",
        DATASET_ROOT / "val" / "labels",
    ]

    if force_reset and DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
        
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    return DATASET_ROOT

def ensure_base_directories():
    required_dirs = [
        ASSETS_PATH / "weights",
        TRAIN_DIR / "config",
        THUMBNAILS_DIR
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

# Run base setup on import
ensure_base_directories()
