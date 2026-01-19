import cv2
import easyocr
import numpy as np
import re
from rapidfuzz import process, utils
from .schema import TradeLayout
from proofreader.core.config import FUZZY_MATCH_CONFIDENCE_THRESHOLD, OCR_LANGUAGES, OCR_USE_GPU

class OCRReader:
    def __init__(self, item_list, languages=OCR_LANGUAGES, gpu=OCR_USE_GPU):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.item_names = [item["name"] for item in item_list]

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

    def process_layout(self, image: np.ndarray, layout: TradeLayout, skip_if=None):
        all_items = layout.outgoing.items + layout.incoming.items
        crops = []
        target_refs = []
        STD_H = 64 

        for item in all_items:
            if skip_if and skip_if(item):
                continue

            if item.name_box:
                x1, y1, x2, y2 = item.name_box.coords
                crop = image[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    new_w = int(w * (STD_H / h))
                    resized = cv2.resize(gray, (new_w, STD_H), interpolation=cv2.INTER_LINEAR)
                    crops.append(resized)
                    target_refs.append({'type': 'item', 'obj': item})
        
        for side in [layout.outgoing, layout.incoming]:
            if side.robux and side.robux.value_box:
                x1, y1, x2, y2 = side.robux.value_box.coords
                crop = image[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    h, w = gray.shape
                    new_w = int(w * (STD_H / h))
                    resized = cv2.resize(gray, (new_w, STD_H), interpolation=cv2.INTER_LINEAR)
                    crops.append(resized)
                    target_refs.append({'type': 'robux', 'obj': side.robux})

        if not crops:
            return
        
        max_w = max(c.shape[1] for c in crops)
        padded_crops = [cv2.copyMakeBorder(c, 0, 0, 0, max_w - c.shape[1], cv2.BORDER_CONSTANT, value=0) for c in crops]

        batch_results = self.reader.readtext_batched(padded_crops, batch_size=len(padded_crops))

        for i, res in enumerate(batch_results):
            raw_text = " ".join([text_info[1] for text_info in res]).strip()
            conf = np.mean([text_info[2] for text_info in res]) if res else 0.0
            
            target = target_refs[i]
            if target['type'] == 'item':
                target['obj'].text_name = raw_text
                target['obj'].text_conf = float(conf)
            else:
                target['obj'].value = self._clean_robux_text(raw_text)
