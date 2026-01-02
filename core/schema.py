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
    id: int = 0
    container_box: Optional[Box] = None
    thumb_box: Optional[Box] = None
    name_box: Optional[Box] = None

@dataclass
class ResolvedRobux:
    value: int = 0
    container_box: Optional[Box] = None
    value_box: Optional[Box] = None

@dataclass
class TradeSide:
    items: List[ResolvedItem] = field(default_factory=list)
    robux: Optional[ResolvedRobux] = None

@dataclass
class TradeLayout:
    outgoing: TradeSide = field(default_factory=TradeSide)
    incoming: TradeSide = field(default_factory=TradeSide)
