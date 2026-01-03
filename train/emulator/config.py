import shutil
from pathlib import Path

EMULATOR_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = EMULATOR_DIR / "templates"

AUGMENTER_PATH = EMULATOR_DIR / "augmenter.js"
DEFAULT_TEMPLATE = TEMPLATES_DIR / "trade_ui.html"

DATASET_ROOT = EMULATOR_DIR.parent / "dataset"

def setup_dataset_directories():
    dirs = [
        DATASET_ROOT / "train" / "images",
        DATASET_ROOT / "train" / "labels",
        DATASET_ROOT / "val" / "images",
        DATASET_ROOT / "val" / "labels",
    ]

    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
        
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    return DATASET_ROOT

if TEMPLATES_DIR.exists():
    TEMPLATE_FILES = [
        str(f.resolve())
        for f in TEMPLATES_DIR.iterdir()
        if f.is_file() and f.name != ".gitkeep"
    ]
else:
    TEMPLATE_FILES = []

def ensure_emulator_paths():
    return setup_dataset_directories()
