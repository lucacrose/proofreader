import easyocr
from core.schema import Box, TradeLayout

class OCRReader:
    def __init__(self, languages=['en'], gpu=True):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def process_layout(self, image_path: str, layout: TradeLayout):
        return
