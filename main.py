from core.detector import TradeDetector
from core.resolver import SpatialResolver
from core.ocr import OCRReader

detector = TradeDetector("assets/weights/yolo_v1.pt")
resolver = SpatialResolver()
reader = OCRReader()

boxes = detector.detect("test.png")
trade_layout = resolver.resolve(boxes)

print(boxes)
print(trade_layout)

reader.process_layout("test.png", trade_layout)

print(trade_layout)
