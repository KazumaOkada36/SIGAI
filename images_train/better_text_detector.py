"""
Better Text Detection - Combines multiple OCR engines
Handles bold text, overlays, and difficult-to-read text
"""

import cv2
import numpy as np
from PIL import Image
import easyocr
import re

class BetterTextDetector:
    """
    Improved text detection using preprocessing and multiple approaches
    """
    
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    def detect_text(self, image_path: str) -> dict:
        """
        Detect text using improved preprocessing
        
        Returns:
            {
                'all_text': str,  # All detected text combined
                'text_blocks': list,  # Individual text blocks
                'has_urgency': bool,
                'has_price': bool,
                'has_discount': bool,
                'has_cta': bool
            }
        """
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return self._empty_result()
        
        # Try multiple preprocessing approaches
        all_detected_text = []
        
        # 1. Original image
        text1 = self._extract_with_easyocr(img)
        all_detected_text.extend(text1)
        
        # 2. Grayscale + contrast enhancement (helps with bold text)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        text2 = self._extract_with_easyocr(contrast)
        all_detected_text.extend(text2)
        
        # 3. Binary threshold (helps with white text on dark backgrounds)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text3 = self._extract_with_easyocr(binary)
        all_detected_text.extend(text3)
        
        # 4. Inverted (helps with light backgrounds)
        inverted = cv2.bitwise_not(gray)
        text4 = self._extract_with_easyocr(inverted)
        all_detected_text.extend(text4)
        
        # Combine and deduplicate
        all_text = " ".join(all_detected_text).upper()
        unique_words = set(all_text.split())
        
        # Analyze the detected text
        return {
            'all_text': all_text,
            'text_blocks': list(unique_words),
            'has_urgency': self._detect_urgency(all_text),
            'has_price': self._detect_price(all_text),
            'has_discount': self._detect_discount(all_text),
            'has_cta': self._detect_cta(all_text),
            'urgency_score': self._count_urgency_words(all_text)
        }
    
    def _extract_with_easyocr(self, img):
        """Extract text using EasyOCR"""
        try:
            # Handle different image types
            if len(img.shape) == 2:  # Grayscale
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            results = self.easyocr_reader.readtext(img_rgb)
            return [text for (_, text, _) in results if text.strip()]
        except:
            return []
    
    def _detect_urgency(self, text: str) -> bool:
        """Check for urgency language"""
        urgency_keywords = [
            'NOW', 'TODAY', 'TOMORROW', 'HOURS', 'MINUTES', 'LIMITED',
            'HURRY', 'LAST CHANCE', 'EXPIRES', 'ENDING', 'ONLY', 'LEFT',
            'QUICK', 'FAST', 'IMMEDIATELY', 'DON\'T MISS', 'BEFORE'
        ]
        return any(keyword in text for keyword in urgency_keywords)
    
    def _detect_price(self, text: str) -> bool:
        """Check for pricing"""
        # Look for dollar signs, cents, or common price patterns
        return bool(re.search(r'[\$\¢]|\d+\.\d{2}|FREE|\bFREE\b', text))
    
    def _detect_discount(self, text: str) -> bool:
        """Check for discounts"""
        discount_keywords = [
            'OFF', 'SAVE', 'DISCOUNT', 'DEAL', 'SALE', '%', 'PERCENT',
            'CLEARANCE', 'SPECIAL', 'PROMO', 'COUPON'
        ]
        return any(keyword in text for keyword in discount_keywords)
    
    def _detect_cta(self, text: str) -> bool:
        """Check for call-to-action"""
        cta_keywords = [
            'SIGN UP', 'JOIN', 'DOWNLOAD', 'GET', 'BUY', 'SHOP',
            'LEARN MORE', 'CLICK', 'TAP', 'START', 'TRY', 'ORDER',
            'CALL', 'VISIT', 'SUBSCRIBE', 'REGISTER', 'APPLY'
        ]
        return any(keyword in text for keyword in cta_keywords)
    
    def _count_urgency_words(self, text: str) -> int:
        """Count urgency indicators"""
        urgency_patterns = [
            r'HOURS', r'MINUTES', r'TODAY', r'TOMORROW', r'NOW',
            r'LIMITED', r'LAST CHANCE', r'EXPIRES', r'ENDING',
            r'ONLY', r'LEFT', r'HURRY', r'QUICK', r'FAST'
        ]
        count = sum(len(re.findall(pattern, text)) for pattern in urgency_patterns)
        return count
    
    def _empty_result(self):
        """Return empty result"""
        return {
            'all_text': '',
            'text_blocks': [],
            'has_urgency': False,
            'has_price': False,
            'has_discount': False,
            'has_cta': False,
            'urgency_score': 0
        }


# Test it
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = BetterTextDetector()
        result = detector.detect_text(sys.argv[1])
        print("=== DETECTED TEXT ===")
        print(result['all_text'][:500])
        print("\n=== ANALYSIS ===")
        print(f"Has urgency: {result['has_urgency']}")
        print(f"Has price: {result['has_price']}")
        print(f"Has discount: {result['has_discount']}")
        print(f"Has CTA: {result['has_cta']}")
        print(f"Urgency score: {result['urgency_score']}")