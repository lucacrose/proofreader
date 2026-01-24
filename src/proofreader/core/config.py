import torch
import shutil
from pathlib import Path

# --- BASE PATHS ---
# Resolves to the 'proofreader' root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_URL = "https://github.com/lucacrose/proofreader"
VERSION_TAG = "v1.2.0"

# --- ASSETS & MODELS ---
ASSETS_PATH = BASE_DIR / "assets"
MODEL_PATH = ASSETS_PATH / "weights" / "yolo.pt"
DB_PATH = ASSETS_PATH / "item_database.json"
CACHE_PATH = ASSETS_PATH / "item_embeddings_bank.pt"
THUMBNAILS_DIR = ASSETS_PATH / "thumbnails"
TRAIN_THUMBNAILS_DIR = ASSETS_PATH / "train_data"
CLASS_MAP_PATH = ASSETS_PATH / "class_mapping.json"
CLIP_BEST_PATH = ASSETS_PATH / "weights" / "clip.pt"

# --- TRAINING & EMULATOR ---
TRAIN_DIR = BASE_DIR / "proofreader" / "train"
DATA_YAML_PATH = TRAIN_DIR / "config" / "data.yaml"
DATASET_ROOT = TRAIN_DIR / "dataset"

EMULATOR_DIR = TRAIN_DIR / "emulator"
TEMPLATES_DIR = EMULATOR_DIR / "templates"
BACKGROUNDS_DIR = EMULATOR_DIR / "backgrounds"
AUGMENTER_PATH = EMULATOR_DIR / "augmenter.js"
DEFAULT_TEMPLATE = TEMPLATES_DIR / "trade_ui.html"

# --- HYPERPARAMETERS (Training Settings) ---
TRAINING_CONFIG = {
    "epochs": 240,             # Number of times the model sees the whole dataset
    "batch_size": 16,          # Number of images processed at once
    "img_size": 640,           # Standard YOLO resolution
    "patience": 20,            # Stop early if no improvement for 20 epochs
    "close_mosaic_epochs": 32  # Disable mosaic augmentation for the last N epochs
}

# --- AUGMENTER PROBABILITIES AND GENERATOR SETTINGS ---
AUGMENTER_CONFIG = {
    "name": {
        "space_chance": 0.15,           # Chance a character in a randomly generated name is a space
        "double_spacing": False,        # Whether consecutive spaces are allowed in names
    },
    "background": {
        "chance": 0.75,                 # Chance the page background is set to a random image
    },
    "recolor": {
        "chance": 0.75,                 # Chance container background and text are recolored and opacity-adjusted
        "container_min_opacity": 0.3,   # Minimum opacity for container background
        "container_max_opacity": 1.0,   # Maximum opacity for container background
        "text_min_opacity": 0.95,       # Minimum opacity for text inside container
        "text_max_opacity": 1.0,        # Maximum opacity for text inside container
    },
    "cards": {
        "name_hide_chance": 0.2,             # Chance the item card name is hidden
        "thumbnail_hide_chance": 0.2,        # Chance the thumbnail is hidden
        "line_height_min": 12,                # Minimum pixels for item name line height
        "line_height_max": 28,                # Maximum pixels for item name line height
        "left_offset_min": 0,                 # Minimum left margin for name/price offset
        "left_offset_max": 24,                # Maximum left margin for name/price offset
        "top_offset_min": 0,                  # Minimum top margin for name/price offset
        "top_offset_max": 12,                 # Maximum top margin for name/price offset
        "duplicate_price_line_chance": 0.35,  # Chance an item card gets an extra price line
        "display_serial_chance": 0.5,         # Chance the limited icon displays a serial number
    },
    "robux_lines": {
        "colon_suffix_chance": 0.5,       # Chance a robux line ends with a colon
        "duplicate_line_chance": 0.3,     # Chance a second robux line will generate for a side
        "hide_chance": 0.25,              # Chance an individual robux line is hidden
    },
    "generator": {
        "aspect_ratio_min": 1.0,           # Minimum allowed aspect ratio (width / height)
        "aspect_ratio_max": 2.4,           # Maximum allowed aspect ratio
        "width_min": 800,                  # Minimum width in pixels
        "width_max": 2560,                 # Maximum width in pixels
        "height_min": 400,                 # Minimum height in pixels (after aspect ratio calculation)
        "height_max": 1600,                # Maximum height in pixels (after aspect ratio calculation)
        "total_images": 1024,              # Total number of images to generate
        "max_workers": 16,                 # Maximum number of parallel workers for generation
        "train_split_fraction": 0.8,       # Fraction of images used for training vs validation
        "empty_trade_chance": 0.09,        # Chance a trade has no items or robux (negative sample)
    }
}

# Robustness Thresholds
FUZZY_MATCH_CONFIDENCE_THRESHOLD = 60.0
CERTAIN_VISUAL_CONF = 0.995

# --- HARDWARE SETTINGS ---
# Automatically detects if a GPU is available for faster training
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

BUILDER_BATCH_SIZE = 32

# OCR Settings
OCR_LANGUAGES = ['en']
OCR_USE_GPU = (DEVICE == "cuda" or DEVICE == "mps")

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
