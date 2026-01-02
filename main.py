from core.detector import TradeDetector
from core.resolver import SpatialResolver

detector = TradeDetector("assets/weights/yolo_v1.pt")
resolver = SpatialResolver()

boxes = detector.detect("test.png")
trade_layout = resolver.resolve(boxes)

print(boxes)
print(trade_layout)
