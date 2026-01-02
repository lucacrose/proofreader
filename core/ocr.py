import cv2
import easyocr
import numpy as np
import re
from core.schema import Box, TradeLayout, TradeSide

class OCRReader:
    def __init__(self, languages=['en'], gpu=True):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def _clean_robux_text(self, raw_text: str) -> int:
        cleaned = raw_text.upper().strip()

        substitutions = {
            ',': '', '.': '', ' ': '',
            'S': '5', 'O': '0', 'I': '1', 
            'L': '1', 'B': '8', 'G': '6'
        }

        for char, sub in substitutions.items():
            cleaned = cleaned.replace(char, sub)
        
        digits = re.findall(r'\d+', cleaned)

        return int("".join(digits)) if digits else 0

    def _get_text_from_box(self, image: np.ndarray, box: Box, is_robux: bool = False) -> str:
        x1, y1, x2, y2 = box.coords

        crop = image[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
        
        if crop.size == 0:
            return ""
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        if is_robux:
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            results = self.reader.readtext(gray, allowlist="0123456789,S ")
        else:
            results = self.reader.readtext(gray)
            
        return " ".join([res[1] for res in results]).strip()

    def process_side(self, image: np.ndarray, side: TradeSide):
        for item in side.items:
            if item.name_box:
                item.name = self._get_text_from_box(image, item.name_box)

        if side.robux and side.robux.value_box:
            raw_val = self._get_text_from_box(image, side.robux.value_box, is_robux=True)
            side.robux.value = self._clean_robux_text(raw_val)

    def process_layout(self, image: str, layout: TradeLayout):
        self.process_side(image, layout.outgoing)
        self.process_side(image, layout.incoming)
