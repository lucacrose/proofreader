import cv2
import easyocr
import numpy as np
import re
from rapidfuzz import process, utils
from .schema import Box, TradeLayout, TradeSide
from proofreader.core.config import FUZZY_MATCH_CONFIDENCE_THRESHOLD, OCR_LANGUAGES, OCR_USE_GPU

class OCRReader:
    def __init__(self, item_list, languages=OCR_LANGUAGES, gpu=OCR_USE_GPU):
        self.reader = easyocr.Reader(languages, gpu=gpu)

        self.item_names = []

        for item in item_list:
            self.item_names.append(item["name"])

    def _fuzzy_match_name(self, raw_text: str, threshold: float = FUZZY_MATCH_CONFIDENCE_THRESHOLD) -> str:
        if not raw_text or len(raw_text) < 2:
            return raw_text
        
        match = process.extractOne(
            raw_text, 
            self.item_names, 
            processor=utils.default_process
        )

        if match and match[1] >= threshold:
            return match[0]
        
        return raw_text

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
                raw_name = self._get_text_from_box(image, item.name_box)
                item.name = self._fuzzy_match_name(raw_name)

        if side.robux and side.robux.value_box:
            raw_val = self._get_text_from_box(image, side.robux.value_box, is_robux=True)
            side.robux.value = self._clean_robux_text(raw_val)
    
    def process_layout(self, image: np.ndarray, layout: TradeLayout):
        all_items = layout.outgoing.items + layout.incoming.items
        crops = []
        target_refs = []

        # Define a standard height for the batch
        # This solves the "inhomogeneous shape" error
        STD_H = 64 

        for item in all_items:
            if item.name_box:
                x1, y1, x2, y2 = item.name_box.coords
                crop = image[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to standard height while maintaining aspect ratio
                    h, w = gray.shape
                    new_w = int(w * (STD_H / h))
                    resized = cv2.resize(gray, (new_w, STD_H), interpolation=cv2.INTER_CUBIC)
                    
                    crops.append(resized)
                    target_refs.append({'type': 'item', 'obj': item})

        for side in [layout.outgoing, layout.incoming]:
            if side.robux and side.robux.value_box:
                x1, y1, x2, y2 = side.robux.value_box.coords
                crop = image[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    
                    # Resize Robux boxes to the same standard height
                    h, w = gray.shape
                    new_w = int(w * (STD_H / h))
                    resized = cv2.resize(gray, (new_w, STD_H), interpolation=cv2.INTER_CUBIC)
                    
                    crops.append(resized)
                    target_refs.append({'type': 'robux', 'obj': side.robux})

        if not crops:
            return

        # EasyOCR's readtext_batched still needs them to be exactly the same size 
        # to stack into a 4D Tensor. We will pad the widths to match the widest crop.
        max_w = max(c.shape[1] for c in crops)
        padded_crops = []
        for c in crops:
            h, w = c.shape
            # Pad the right side with the mean color or black
            padded = cv2.copyMakeBorder(c, 0, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=0)
            padded_crops.append(padded)

        # Now all images in padded_crops are (STD_H, max_w)
        batch_results = self.reader.readtext_batched(padded_crops, batch_size=len(padded_crops))

        for i, res in enumerate(batch_results):
            raw_text = " ".join([text_info[1] for text_info in res]).strip()
            target = target_refs[i]
            if target['type'] == 'item':
                target['obj'].name = self._fuzzy_match_name(raw_text)
            else:
                target['obj'].value = self._clean_robux_text(raw_text)
