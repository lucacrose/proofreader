from typing import List
from core.schema import Box, ResolvedItem, TradeLayout

class SpatialResolver:
    def __init__(self):
        pass

    def get_center(self, box: Box):
        x1, y1, x2, y2 = box.coords
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    def is_contained(self, child: Box, parent: Box, margin: int = 20) -> bool:
        cx1, cy1, cx2, cy2 = child.coords
        px1, py1, px2, py2 = parent.coords
        
        return (cx1 >= px1 - margin and 
                cy1 >= py1 - margin and 
                cx2 <= px2 + margin and 
                cy2 <= py2 + margin)

    def resolve(self, all_boxes: List[Box]) -> TradeLayout:
        layout = TradeLayout()

        cards = [b for b in all_boxes if b.label == "item_card"]
        robux_lines = [b for b in all_boxes if b.label == "robux_line"]
        
        names = [b for b in all_boxes if b.label == "item_name"]
        thumbs = [b for b in all_boxes if b.label == "item_thumb"]
        values = [b for b in all_boxes if b.label == "robux_value"]

        parents = cards + robux_lines
        if not parents:
            return layout

        y_centers = sorted([self.get_center(p)[1] for p in parents])
        
        if len(y_centers) > 1:
            max_gap = -1
            gap_index = 0

            for i in range(len(y_centers) - 1):
                gap = y_centers[i + 1] - y_centers[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_index = i + 1
            
            first_bottom_parent = next(p for p in parents if self.get_center(p)[1] == y_centers[gap_index])
            split_y = first_bottom_parent.coords[1] - 10
        else:
            split_y = y_centers[0] + 100

        for card in cards:
            item = ResolvedItem(container_box=card)
            item.name_box = next((n for n in names if self.is_contained(n, card)), None)
            item.thumb_box = next((t for t in thumbs if self.is_contained(t, card)), None)

            if self.get_center(card)[1] < split_y:
                layout.outgoing_items.append(item)
            else:
                layout.incoming_items.append(item)

        for line in robux_lines:
            val_box = next((v for v in values if self.is_contained(v, line)), None)
            
            if val_box:
                if self.get_center(line)[1] < split_y:
                    layout.outgoing_robux_box = val_box 
                else:
                    layout.incoming_robux_box = val_box

        return layout
