# Proofreader 🔍
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![YOLOv11](https://img.shields.io/badge/model-YOLOv11-green.svg)
![License](https://img.shields.io/badge/license-MIT-red.svg)

## Example
| Input Image | Detected UI Elements |
|--------|-------|
| ![](./docs/assets/trade_before.png) | ![](./docs/assets/trade_after.png) |

### Output
When a trade image is processed, the engine returns a structured JSON object:

```json
{
    "outgoing": {
        "item_count": 4,
        "robux_value": 0,
        "items": [
            {
                "id": 1031429,
                "name": "Domino Crown"
            },
            {
                "id": 72082328,
                "name": "Red Sparkle Time Fedora"
            },
            {
                "id": 124730194,
                "name": "Blackvalk"
            },
            {
                "id": 16652251,
                "name": "Red Tango"
            }
        ]
    },
    "incoming": {
        "item_count": 2,
        "robux_value": 1048576,
        "items": [
            {
                "id": 21070012,
                "name": "Dominus Empyreus"
            },
            {
                "id": 22850569,
                "name": "Red Bandana of SQL Injection"
            }
        ]
    }
}
```

### 💻 Quick Start
You can process any trade image with just a few lines of code:

```python
import proofreader

# Analyze the image
data = proofreader.get_trade_data("test.png")

# Print the result
print(data)
```

### Installation
For quick installation, simply run:

```bash
pip install rbx-proofreader
```
