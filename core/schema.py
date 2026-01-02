from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Box:
    coords: Tuple[int, int, int, int]
    label: str
    confidence: float

@dataclass
class ResolvedItem:
    name: str = "Unknown"
    value: int = 0
    container_box: Optional[Box] = None
    thumb_box: Optional[Box] = None
    name_box: Optional[Box] = None
    value_box: Optional[Box] = None

@dataclass
class TradeLayout:
    outgoing_items: List[ResolvedItem] = field(default_factory=list)
    incoming_items: List[ResolvedItem] = field(default_factory=list)
    incoming_robux_box: int = 0
    outgoing_robux_box: int = 0
