from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_PATH = BASE_DIR / "assets"
MODEL_PATH = ASSETS_PATH / "weights" / "yolo_v1.pt"
DB_PATH = ASSETS_PATH / "db.json"
CACHE_PATH = ASSETS_PATH / "embedding_bank.pt"

TRAIN_DIR = BASE_DIR / "train"
ASSETS_DIR = ASSETS_PATH / "thumbnails"
DATA_YAML_PATH = TRAIN_DIR / "config" / "data.yaml"

def ensure_directories():
    required_dirs = [
        ASSETS_PATH,
        ASSETS_PATH / "weights",
        TRAIN_DIR / "config",
        TRAIN_DIR / "dataset"
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()
