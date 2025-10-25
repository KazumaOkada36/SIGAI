"""
IMAGE AD FEATURE EXTRACTOR - Your Module
Clean interface for the team integration
Input: Image file path
Output: Dictionary of 40+ features
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings('ignore')

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False
    print("⚠️  Warning: easyocr not installed. Install with: pip install easyocr")


class ImageAdAnalyzer:
    """
    Main class for extracting features from IMAGE advertisements.
    Usage:
        analyzer = ImageAdAnalyzer()
        features = analyzer.analyze(image_path)
    """
    
    def __init__(self):
        """Initialize all models (only once, then reuse)"""
        print("🔧 Loading models...")
        
        # Load CLIP for semantic understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        # Load OCR for text extraction
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except:
                self.ocr_reader = None
        else:
            self.ocr_reader = None
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("✅ Models loaded successfully!\n")
    
    def analyze(self, image_path: str) -> dict:
        """
        Main function - analyzes an image ad and returns all features.
        
        Args:
            image_path: Path to image file (.png, .jpg, .jpeg)
        
        Returns:
            Dictionary with 40+ features
        """
        
        print(f"📸 Analyzing: {image_path}")
        
        # Load image
        try:
            img_cv = cv2.imread(image_path)
            img_pil = Image.open(image_path).convert('RGB')
            
            if img_cv is None:
                raise ValueError(f"Could not load image: {image_path}")
        except Exception as e:
            return {'error': str(e)}
        
        # Initialize features dictionary
        features = {'image_path': image_path}
        
        # Extract all feature categories
        features.update(self._extract_visual_features(img_cv, img_pil))
        features.update(self._extract_text_features(img_cv))
        features.update(self._extract_engagement_features(img_cv))
        features.update(self._extract_novel_features(img_cv, features))
        
        print(f"✅ Extracted {len(features)} features\n")
        
        return features
    
    # ============================================
    # VISUAL FEATURES (13 features)
    # ============================================
    
    def _extract_visual_features(self, img_cv, img_pil):
        """Extract visual foundation features"""
        
        features = {}
        
        # 1. COLOR FEATURES
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Brightness
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        features['avg_brightness'] = round(float(np.mean(gray)), 2)
        
        # Saturation
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        features['avg_saturation'] = round(float(np.mean(hsv[:, :, 1])), 2)
        
        # Color contrast
        features['color_contrast'] = round(float(np.std(gray)), 2)
        
        # Color diversity
        pixels = img_rgb.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        features['color_diversity_score'] = round(len(unique_colors) / len(pixels) * 100, 2)
        
        # 2. COMPOSITION FEATURES
        
        # Visual complexity (edge density)
        edges = cv2.Canny(gray, 50, 150)
        features['visual_complexity'] = round(float(np.sum(edges > 0) / edges.size * 100), 2)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        features['face_count'] = len(faces)
        
        if len(faces) > 0:
            total_face_area = sum([w * h for (x, y, w, h) in faces])
            features['face_to_frame_ratio'] = round(total_face_area / (img_cv.shape[0] * img_cv.shape[1]) * 100, 2)
        else:
            features['face_to_frame_ratio'] = 0.0
        
        # Image sharpness
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['image_sharpness'] = round(min(blur_score / 1000, 1.0), 3)
        
        # 3. CLIP SEMANTIC FEATURES (zero-shot classification)
        with torch.no_grad():
            concepts = {
                'product_focus': 'a product being displayed prominently',
                'people_present': 'people or faces in the image',
                'luxury_aesthetic': 'luxury, premium, elegant, high-end',
                'playful_tone': 'playful, fun, colorful, energetic',
                'minimal_design': 'minimal, simple, clean design'
            }
            
            inputs = self.clip_processor(images=img_pil, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            text_inputs = self.clip_processor(
                text=list(concepts.values()),
                return_tensors="pt",
                padding=True
            )
            text_features = self.clip_model.get_text_features(**text_inputs)
            
            # Normalize and compute similarities
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)
            probs = torch.softmax(similarities * 100, dim=0)
            
            for i, (key, _) in enumerate(concepts.items()):
                features[f'clip_{key}_score'] = round(float(probs[i].item()), 3)
        
        return features
    
    # ============================================
    # TEXT & CTA FEATURES (11 features)
    # ============================================
    
    def _extract_text_features(self, img_cv):
        """Extract text and CTA features using OCR"""
        
        features = {}
        
        # Run OCR
        detected_text = []
        if self.ocr_reader:
            try:
                results = self.ocr_reader.readtext(img_cv)
                detected_text = [(text, bbox) for bbox, text, conf in results if conf > 0.3]
            except:
                pass
        
        # Text presence
        features['text_element_count'] = len(detected_text)
        
        all_text = ' '.join([text for text, _ in detected_text]).lower()
        features['total_text_length'] = len(all_text)
        
        # Text density (% of image covered by text)
        if detected_text:
            total_text_area = 0
            for text, bbox in detected_text:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        width = abs(bbox[1][0] - bbox[0][0])
                        height = abs(bbox[2][1] - bbox[1][1])
                        total_text_area += width * height
                    except:
                        pass
            
            image_area = img_cv.shape[0] * img_cv.shape[1]
            features['text_density_pct'] = round((total_text_area / image_area * 100) if image_area > 0 else 0, 2)
            
            # Average text size
            sizes = []
            for text, bbox in detected_text:
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        height = abs(bbox[2][1] - bbox[1][1])
                        sizes.append(height)
                    except:
                        pass
            features['avg_text_size_px'] = round(float(np.mean(sizes)), 1) if sizes else 0.0
        else:
            features['text_density_pct'] = 0.0
            features['avg_text_size_px'] = 0.0
        
        # CTA detection
        cta_keywords = [
            'buy', 'shop', 'get', 'download', 'install', 'try', 'start',
            'learn', 'sign up', 'subscribe', 'join', 'order', 'book',
            'call', 'click', 'tap', 'discover', 'free trial', 'buy now'
        ]
        features['cta_keyword_count'] = sum(1 for kw in cta_keywords if kw in all_text)
        features['has_cta'] = features['cta_keyword_count'] > 0
        
        # Button-like text (ALL CAPS)
        import re
        all_text_original = ' '.join([text for text, _ in detected_text])
        button_pattern = r'\b[A-Z]{2,}\b'
        features['button_text_count'] = len(re.findall(button_pattern, all_text_original))
        
        # Price detection
        price_pattern = r'[\$€£¥]\s*\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*[\$€£¥]'
        prices = re.findall(price_pattern, all_text_original)
        features['has_price'] = len(prices) > 0
        features['price_count'] = len(prices)
        
        # Discount detection
        discount_pattern = r'\d+%\s*(?:off|save)|(?:save|discount)\s*\d+%'
        features['has_discount'] = bool(re.search(discount_pattern, all_text))
        
        # Urgency language
        urgency_keywords = [
            'now', 'today', 'limited', 'hurry', 'last chance', 'ending soon',
            'don\'t miss', 'act fast', 'expires', 'only', 'exclusive'
        ]
        features['urgency_language_score'] = sum(1 for kw in urgency_keywords if kw in all_text)
        
        return features
    
    # ============================================
    # ENGAGEMENT FEATURES (11 features)
    # ============================================
    
    def _extract_engagement_features(self, img_cv):
        """Extract engagement prediction features"""
        
        features = {}
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h, w = img_cv.shape[:2]
        
        # ATTENTION FEATURES
        
        # Color pop (high saturation ratio)
        features['attention_color_pop'] = round(float(np.sum(hsv[:, :, 1] > 150) / hsv[:, :, 1].size * 100), 2)
        
        # Contrast ratio
        dark_pixels = np.sum(gray < 85) / gray.size
        light_pixels = np.sum(gray > 170) / gray.size
        features['attention_contrast_ratio'] = round(float(abs(dark_pixels - light_pixels)), 3)
        
        # Center focus
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        edge_center = cv2.Canny(center_region, 50, 150)
        features['center_focus_score'] = round(float(np.sum(edge_center > 0) / edge_center.size * 100), 2)
        
        # Rule of thirds
        edges = cv2.Canny(gray, 50, 150)
        third_h, third_w = h // 3, w // 3
        thirds_points = [(third_w, third_h), (2*third_w, third_h), (third_w, 2*third_h), (2*third_w, 2*third_h)]
        thirds_score = 0
        for x, y in thirds_points:
            region = edges[max(0,y-20):min(h,y+20), max(0,x-20):min(w,x+20)]
            if region.size > 0 and np.sum(region > 0) > 50:
                thirds_score += 1
        features['rule_of_thirds_score'] = thirds_score
        
        # MOBILE OPTIMIZATION
        aspect_ratio = w / h
        features['mobile_aspect_ratio'] = 1 if 0.5 < aspect_ratio < 0.7 else 0
        
        total_pixels = h * w
        features['hd_quality'] = 1 if total_pixels >= 1280 * 720 else 0
        
        edge_density = np.sum(edges > 0) / edges.size
        if total_pixels < 640 * 480 and edge_density > 0.1:
            features['mobile_readability_score'] = 0.3
        elif total_pixels >= 1280 * 720:
            features['mobile_readability_score'] = 1.0
        else:
            features['mobile_readability_score'] = 0.7
        
        features['mobile_simplicity_score'] = round(max(0, 1 - edge_density * 5), 2)
        
        # VISUAL APPEAL
        
        # Color harmony
        hue_values = hsv[:, :, 0].flatten()
        hue_hist, _ = np.histogram(hue_values, bins=12)
        hue_entropy = -np.sum((hue_hist / hue_hist.sum()) * np.log2((hue_hist / hue_hist.sum()) + 1e-10))
        features['color_harmony_score'] = round(min(hue_entropy / 3.5, 1.0), 3)
        
        # Visual balance (symmetry)
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        if left_half.shape != right_half.shape:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        diff = cv2.absdiff(left_half, right_half)
        features['visual_balance_score'] = round(1 - (float(np.mean(diff)) / 255), 3)
        
        # Professional look
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(blur_score / 1000, 1.0)
        contrast = gray.std() / 128
        features['professional_look_score'] = round((sharpness * 0.4 + contrast * 0.3 + features['visual_balance_score'] * 0.3), 3)
        
        return features
    
    # ============================================
    # NOVEL FEATURES (8 features) - YOUR SECRET SAUCE
    # ============================================
    
    def _extract_novel_features(self, img_cv, existing_features):
        """Extract novel, creative features"""
        
        features = {}
        
        # 1. SCROLLABILITY SCORE - Will users stop scrolling?
        score = 0.0
        score += (existing_features.get('attention_color_pop', 0) / 100) * 0.3
        score += (existing_features.get('center_focus_score', 0) / 100) * 0.25
        face_score = min(existing_features.get('face_count', 0) / 3, 1.0)
        score += face_score * 0.2
        score += existing_features.get('attention_contrast_ratio', 0) * 0.15
        complexity = existing_features.get('visual_complexity', 50)
        clutter_score = 1.0 if 30 <= complexity <= 60 else 0.5
        score += clutter_score * 0.1
        features['scrollability_score'] = round(score, 3)
        
        # 2. CURIOSITY GAP SCORE - Does it create questions?
        score = 0.0
        text_count = existing_features.get('text_element_count', 0)
        if text_count == 0:
            score += 0.3
        elif 1 <= text_count <= 3:
            score += 0.7
        elif 4 <= text_count <= 7:
            score += 0.4
        else:
            score += 0.1
        if not existing_features.get('has_cta', False):
            score += 0.2
        if not existing_features.get('has_price', False):
            score += 0.1
        features['curiosity_gap_score'] = round(score, 3)
        
        # 3. VALUE PROP CLARITY - Can you understand in 2 seconds?
        score = 1.0
        if text_count == 0:
            score -= 0.4
        elif text_count > 10:
            score -= 0.2
        product_focus = existing_features.get('clip_product_focus_score', 0)
        if product_focus < 0.3:
            score -= 0.3
        if existing_features.get('has_cta', False):
            score += 0.2
        else:
            score -= 0.1
        if complexity > 70:
            score -= 0.2
        features['value_prop_clarity'] = round(max(0, min(1, score)), 3)
        
        # 4. INFORMATION DENSITY - sparse/optimal/overloaded?
        density_score = (text_count * 5) + complexity
        if density_score < 40:
            features['information_density'] = "sparse"
        elif density_score < 120:
            features['information_density'] = "optimal"
        else:
            features['information_density'] = "overloaded"
        
        # 5. FIRST IMPRESSION SCORE - How striking is it?
        score = 0.0
        score += (existing_features.get('color_contrast', 0) / 100) * 0.35
        score += (existing_features.get('avg_saturation', 0) / 255) * 0.25
        score += existing_features.get('professional_look_score', 0) * 0.2
        score += (existing_features.get('rule_of_thirds_score', 0) / 4) * 0.2
        features['first_impression_score'] = round(score, 3)
        
        # 6. BRAND vs PRODUCT BALANCE
        product_score = existing_features.get('clip_product_focus_score', 0)
        luxury_score = existing_features.get('clip_luxury_aesthetic_score', 0)
        minimal_score = existing_features.get('clip_minimal_design_score', 0)
        brand_signals = (luxury_score * 0.4) + (minimal_score * 0.4) + ((1 - product_score) * 0.2)
        
        if product_score > 0.6:
            features['brand_product_balance'] = "product_focused"
        elif brand_signals > 0.5:
            features['brand_product_balance'] = "brand_focused"
        else:
            features['brand_product_balance'] = "balanced"
        
        # 7. CTA PROMINENCE - How visible is the call-to-action?
        cta_score = 0.0
        if existing_features.get('has_cta', False):
            cta_score += 0.4
            if existing_features.get('button_text_count', 0) > 0:
                cta_score += 0.3
            if existing_features.get('avg_text_size_px', 0) > 30:
                cta_score += 0.3
        features['cta_prominence_score'] = round(cta_score, 2)
        
        # 8. EMOTIONAL APPEAL - Does it have emotional elements?
        emotional_score = 0.0
        if existing_features.get('face_count', 0) > 0:
            emotional_score += 0.4
        if existing_features.get('clip_people_present_score', 0) > 0.5:
            emotional_score += 0.3
        if existing_features.get('clip_playful_tone_score', 0) > 0.5:
            emotional_score += 0.3
        features['emotional_appeal_score'] = round(emotional_score, 2)
        
        return features


# ============================================
# SIMPLE USAGE EXAMPLE
# ============================================

def main():
    """Example usage"""
    
    # Initialize analyzer (do this once)
    analyzer = ImageAdAnalyzer()
    
    # Analyze a single image
    image_path = "path/to/your/image.png"
    features = analyzer.analyze(image_path)
    
    # Print results
    print("\n" + "="*60)
    print("EXTRACTED FEATURES:")
    print("="*60)
    for key, value in features.items():
        print(f"{key}: {value}")
    
    # Or process multiple images
    import glob
    image_folder = "ads/"  # Your folder with images
    all_features = []
    
    for img_path in glob.glob(f"{image_folder}*.png") + glob.glob(f"{image_folder}*.jpg"):
        features = analyzer.analyze(img_path)
        all_features.append(features)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(all_features)
    df.to_csv("image_ad_features.csv", index=False)
    print(f"\n✅ Saved {len(all_features)} image features to image_ad_features.csv")


if __name__ == "__main__":
    main()