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

    visual_id: int = -1
    visual_conf: float = 0

    text_name: str = ""
    text_conf: float = 0

    _finalized: bool = False

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

    def to_dict(self, row_tolerance=16) -> dict:
        def serialize_side(side: TradeSide):
            robux_val = side.robux.value if side.robux else 0
            
            if not side.items:
                return {"item_count": 0, "robux_value": robux_val, "items": []}
            
            raw_items = [it for it in side.items if it.container_box]
            raw_items.sort(key=lambda x: x.container_box.coords[1])

            rows = []
            if raw_items:
                current_row = [raw_items[0]]
                for i in range(1, len(raw_items)):
                    prev_y = current_row[-1].container_box.coords[1]
                    curr_y = raw_items[i].container_box.coords[1]
                    
                    if abs(curr_y - prev_y) < row_tolerance:
                        current_row.append(raw_items[i])
                    else:
                        rows.append(current_row)
                        current_row = [raw_items[i]]
                rows.append(current_row)
            
            final_sorted = []
            for row in rows:
                row.sort(key=lambda x: x.container_box.coords[0])
                final_sorted.extend(row)

            return {
                "item_count": len(side.items),
                "robux_value": robux_val,
                "items": [
                    {"id": item.id, "name": item.name} 
                    for item in final_sorted
                ]
            }

        return {
            "outgoing": serialize_side(self.outgoing),
            "incoming": serialize_side(self.incoming)
        }
