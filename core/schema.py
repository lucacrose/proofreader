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
    container_box: Optional[Box] = None
    thumb_box: Optional[Box] = None
    name_box: Optional[Box] = None

@dataclass
class ResolvedRobux:
    value: int = 0
    value_box: Optional[Box] = None

@dataclass
class TradeLayout:
    outgoing_items: List[ResolvedItem] = field(default_factory=list)
    incoming_items: List[ResolvedItem] = field(default_factory=list)
    incoming_robux: ResolvedRobux = field(default_factory=ResolvedRobux)
    outgoing_robux: ResolvedRobux = field(default_factory=ResolvedRobux)
