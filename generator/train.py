from ultralytics import YOLO

def train_model():
    # 1. Load a pretrained YOLOv8 Nano model
    # 'n' stands for nano—it's very fast for real-time proofreading
    model = YOLO("yolo11n.pt")

    # 2. Start Training
    model.train(
        data="data.yaml",
        epochs=256,
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

    print("Training complete! Your model is in 'runs/detect/train/weights/best.pt'")

def finish_training():
    # 1. Load your CURRENT best/last checkpoint
    # Replace 'runs/detect/train/weights/last.pt' with your actual path
    model = YOLO("runs/detect/train48/weights/last.pt")

    # 2. Resume with Overrides
    # We set epochs to 225 (giving you ~20 epochs of 'clean' training)
    # We set close_mosaic to 25 to ensure it triggers IMMEDIATELY upon resume
    model.train(
        data="data.yaml",    # You MUST re-specify data when not using resume
        epochs=32,           # Set this to the total number of NEW epochs you want
        close_mosaic=32,     # This will now trigger IMMEDIATELY (since 32/32 = start now)
        patience=20,
        imgsz=640,           # Keep your other settings consistent
        batch=24,
        device=0
    )

    print("Refinement complete! Check 'best.pt' for the final snapped-in weights.")

if __name__ == "__main__":
    finish_training()
