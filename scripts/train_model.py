from proofreader.train.emulator.generator import run_mass_generation
from proofreader.train.yolo_trainer import train_yolo
from proofreader.train.clip_trainer import train_clip

if __name__ == "__main__":
    run_mass_generation()

    train_yolo(0) # Change from 0 -> "cpu" if no CUDA devices

    train_clip()
