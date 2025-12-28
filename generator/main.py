from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"
TEMPLATE_PATH = BASE_DIR / "trade_template.html"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
