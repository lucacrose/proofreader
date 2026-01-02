from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n.pt")

    model.train(
        data="train/config/data.yaml",
        epochs=240,
        imgsz=640,
        device=0,
        plots=True,
        multi_scale=True,

        batch=24,
        patience=48,

        box=7.5,
        dfl=1.5,
        cls=1.5,

        mosaic=0.7,
        mixup=0.05,
        close_mosaic=32,

        overlap_mask=False,
        workers=8
    )

def finish_training(file_path):
    model = YOLO(file_path)

    model.train(
        data="train/config/data.yaml",
        epochs=32,
        close_mosaic=32,
        patience=20,
        imgsz=640,
        batch=24,
        device=0
    )

if __name__ == "__main__":
    train_model()
