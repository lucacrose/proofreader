from core.detector import TradeDetector

detector = TradeDetector("assets/weights/yolo_v1.pt")

boxes = detector.detect("test.png")

print(boxes)