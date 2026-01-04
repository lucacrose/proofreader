# Proofreader 🔍
High-speed Roblox trade analyzer using **YOLOv11**, **CLIP**, and **EasyOCR** for instant item detection and structured JSON output.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![YOLOv11](https://img.shields.io/badge/model-YOLOv11-green.svg)
![License](https://img.shields.io/badge/license-MIT-red.svg)
![PyPI](https://img.shields.io/pypi/v/rbx-proofreader?color=blue&label=PyPI)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

---

## Features
- Detect outgoing and incoming trade items from screenshots
- Outputs structured JSON for automation or analytics
- Supports custom backgrounds and HTML templates
- Trained with YOLOv11 and enhanced with CLIP embeddings
- Easy installation and integration

---

## Example

| Input Image | Detected UI Elements |
| ----------- | ------------------ |
| ![](./docs/assets/trade_before.png) | ![](./docs/assets/trade_after.png) |

**Sample Output JSON:**

```json
{
    "outgoing": {
        "item_count": 4,
        "robux_value": 0,
        "items": [
            {"id": 1031429, "name": "Domino Crown"},
            {"id": 72082328, "name": "Red Sparkle Time Fedora"},
            {"id": 124730194, "name": "Blackvalk"},
            {"id": 16652251, "name": "Red Tango"}
        ]
    },
    "incoming": {
        "item_count": 2,
        "robux_value": 1048576,
        "items": [
            {"id": 21070012, "name": "Dominus Empyreus"},
            {"id": 22850569, "name": "Red Bandana of SQL Injection"}
        ]
    }
}
```

## 💻 Quick Start

```py
import proofreader

# Analyze the image
data = proofreader.get_trade_data("test.png")

# Print the result
print(data)
```

## Installation

### Quick Install (Recommended)

```bash
pip install rbx-proofreader
```

### From Source (Advanced / Custom Training)

**1.** Clone the repository.

**2.** Run `python scripts/setup_items.py` to initialize cache, download thumbnails, and create CLIP embeddings.

**3.** Place background images in: `src/proofreader/train/emulator/backgrounds`. Use continuous numbering: background_0.jpg, background_1.jpg, ...

**4.** Place HTML templates in `src/proofreader/train/emulator/templates`. Include both light and dark theme templates.

**5.** Configure synthetic data generation and YOLO training settings in `src/proofreader/core/config.py`


**6.** Run `python scripts/train_model.py`

> Note: GPU recommended for training. Final model will be saved under `runs/trainX/weights/best.pt`. Rename to `yolo.pt` and move to `src/assets/weights`.

## Tech Stack

- **Python 3.12**

- **YOLOv11** for fast UI detection

- **CLIP** for visual embedding matching

- **EasyOCR** for text extraction

- **NumPy / OpenCV** for image processing

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.