from ultralytics import YOLO
from ..core.config import TRAINING_CONFIG, DATA_YAML_PATH

def train_yolo(device):
    model = YOLO("yolo11n.pt")

    model.train(
        data = DATA_YAML_PATH,
        epochs = TRAINING_CONFIG["epochs"],
        imgsz = TRAINING_CONFIG["img_size"],
        device = device,
        plots = True,
        multi_scale = True,

        batch = TRAINING_CONFIG["batch_size"],
        patience = TRAINING_CONFIG["patience"],

        box = 7.5,
        dfl = 1.5,
        cls = 1.5,

        mosaic = 0.7,
        mixup = 0.05,
        close_mosaic = TRAINING_CONFIG["close_mosaic_epochs"],

        overlap_mask = False,
        workers = 8
    )

def finish_training(file_path, device):
    model = YOLO(file_path)

    model.train(
        data = DATA_YAML_PATH,
        epochs = 32,
        close_mosaic = 32,
        patience = 20,
        imgsz = TRAINING_CONFIG["img_size"],
        batch = TRAINING_CONFIG["batch_size"],
        device = device
    )
