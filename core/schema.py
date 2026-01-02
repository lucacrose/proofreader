from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Box:
    coords: Tuple[int, int, int, int]
    label: str
    confidence: float