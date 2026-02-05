from typing import List
from .schema import Box, ResolvedItem, ResolvedRobux, TradeLayout

class SpatialResolver:
    def __init__(self):
        pass

    def get_iou(self, box1: Box, box2: Box) -> float:
        b1x1, b1y1, b1x2, b1y2 = box1.coords
        b2x1, b2y1, b2x2, b2y2 = box2.coords
        
        ix1, iy1 = max(b1x1, b2x1), max(b1y1, b2y1)
        ix2, iy2 = min(b1x2, b2x2), min(b1y2, b2y2)
        
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        area2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def get_ioa(self, child: Box, parent: Box) -> float:
        cx1, cy1, cx2, cy2 = child.coords
        px1, py1, px2, py2 = parent.coords
        
        ix1, iy1 = max(cx1, px1), max(cy1, py1)
        ix2, iy2 = min(cx2, px2), min(cy2, py2)
        
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        child_area = (cx2 - cx1) * (cy2 - cy1)
        return inter_area / child_area if child_area > 0 else 0

    def resolve(self, all_boxes: List[Box]) -> TradeLayout:
        layout = TradeLayout()

        raw_cards = sorted([b for b in all_boxes if b.label == "item_card"], 
                           key=lambda x: x.confidence, reverse=True)
        
        unique_cards = []
        for card in raw_cards:
            if any(self.get_iou(card, accepted) > 0.5 for accepted in unique_cards):
                continue
            unique_cards.append(card)
        
        robux_lines = [b for b in all_boxes if b.label == "robux_line"]
        names = [b for b in all_boxes if b.label == "item_name"]
        thumbs = [b for b in all_boxes if b.label == "item_thumb"]
        values = [b for b in all_boxes if b.label == "robux_value"]
        header_received = next((b for b in all_boxes if b.label == "received_header"), None)

        if header_received:
            split_y = header_received.coords[1]
        else:
            parents = sorted(unique_cards + robux_lines, key=lambda b: (b.coords[1] + b.coords[3])/2)
            if len(parents) > 1:
                y_centers = [(b.coords[1] + b.coords[3])/2 for b in parents]
                max_gap = 0
                split_y = y_centers[0] + 50 
                for i in range(len(y_centers) - 1):
                    gap = y_centers[i+1] - y_centers[i]
                    if gap > max_gap:
                        max_gap = gap
                        split_y = (y_centers[i] + y_centers[i+1]) / 2
            else:
                split_y = 500
        
        unique_cards.sort(key=lambda b: b.coords[1])

        for card in unique_cards:
            item = ResolvedItem(container_box=card)

            item.name_box = next((n for n in names if self.get_ioa(n, card) > 0.7), None)
            item.thumb_box = next((t for t in thumbs if self.get_ioa(t, card) > 0.7), None)

            if (card.coords[1] + card.coords[3]) / 2 < split_y:
                layout.outgoing.items.append(item)
            else:
                layout.incoming.items.append(item)
        
        for line in robux_lines:
            val_box = next((v for v in values if self.get_ioa(v, line) > 0.5), None)
            if val_box:
                robux_obj = ResolvedRobux(container_box=line, value_box=val_box)
                if (line.coords[1] + line.coords[3]) / 2 < split_y:
                    layout.outgoing.robux = robux_obj
                else:
                    layout.incoming.robux = robux_obj

        return layout
