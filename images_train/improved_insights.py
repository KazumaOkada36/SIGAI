"""
IMPROVED Human-Readable Ad Insights Generator
Actually USES the detected text and visual features properly
"""

from image_analyzer import ImageAdAnalyzer
import re

class ImprovedAdInsightsGenerator:
    """
    Generates human-readable insights that actually use the extracted features correctly
    """
    
    def __init__(self):
        self.analyzer = ImageAdAnalyzer()
    
    def generate_insights(self, image_path: str) -> dict:
        """Generate insights that are actually based on what we detect"""
        
        # Extract features
        features = self.analyzer.analyze(image_path)
        
        if 'error' in features:
            return {'error': features['error']}
        
        # Generate insights using ACTUAL detected content
        insights = {
            'image_path': image_path,
            'summary': self._generate_summary(features),
            'target_audience': self._infer_target_audience(features),
            'product_type': self._infer_product_type(features),
            'emotional_appeal': self._describe_emotional_appeal(features),
            'visual_style': self._describe_visual_style(features),
            'engagement_prediction': self._predict_engagement(features),
            'strengths': self._identify_strengths(features),
            'weaknesses': self._identify_weaknesses(features),
            'recommendations': self._generate_recommendations(features),
            'technical_features': features
        }
        
        return insights
    
    def _generate_summary(self, features: dict) -> str:
        """Generate accurate summary based on actual detected features"""
        
        parts = []
        
        # Check what we ACTUALLY detected
        has_urgency = features.get('urgency_language_score', 0) > 0
        has_price = features.get('has_price', False)
        has_discount = features.get('has_discount', False)
        has_cta = features.get('has_cta', False)
        face_count = features.get('face_count', 0)
        
        # Be specific about what type of ad this is
        if (has_urgency and has_price) or (has_discount and has_cta):
            parts.append("A promotional direct-response advertisement")
        elif has_price or has_discount:
            parts.append("A sales-focused advertisement")
        elif has_cta:
            parts.append("A conversion-focused advertisement")
        else:
            parts.append("An awareness-building advertisement")
        
        # Add people context
        if face_count >= 3:
            parts.append(f"featuring {face_count} people")
        elif face_count > 0:
            parts.append(f"featuring {face_count} person" + ("s" if face_count > 1 else ""))
        
        # Add urgency/CTA context
        if has_urgency:
            parts.append("with time-sensitive messaging")
        if has_cta:
            parts.append("with clear call-to-action")
        
        return ", ".join(parts) + "."
    
    def _infer_target_audience(self, features: dict) -> dict:
        """Infer audience based on ACTUAL features - especially fitness/gym context"""
        
        audience = {
            'primary_demographic': [],
            'characteristics': [],
            'confidence': 'medium',
            'description': ''
        }
        
        # Get key signals
        faces = features.get('face_count', 0)
        people_score = features.get('clip_people_present_score', 0)
        has_urgency = features.get('urgency_language_score', 0) > 2
        has_discount = features.get('has_discount', False)
        has_cta = features.get('has_cta', False)
        
        # Pattern 1: Fitness/gym (people + urgency + discount)
        if faces >= 2 and people_score > 0.6 and (has_urgency or has_discount):
            audience['primary_demographic'].append("fitness-conscious adults (18-35)")
            audience['characteristics'].extend([
                "health and wellness focused",
                "motivated by deals and promotions",
                "action-oriented",
                "values physical activity",
                "budget-conscious gym-goers"
            ])
            audience['confidence'] = 'high'
        
        # Pattern 2: Deal seekers (urgency + discount)
        elif has_urgency and has_discount:
            audience['primary_demographic'].append("deal-conscious consumers (20-45)")
            audience['characteristics'].extend([
                "price-sensitive",
                "urgency-responsive",
                "quick decision-makers",
                "motivated by savings"
            ])
            audience['confidence'] = 'high'
        
        # Pattern 3: Action-oriented (strong CTA + urgency)
        elif has_cta and has_urgency:
            audience['primary_demographic'].append("action-oriented consumers (25-45)")
            audience['characteristics'].extend([
                "responsive to urgency",
                "ready to convert",
                "immediate action takers"
            ])
            audience['confidence'] = 'medium'
        
        # Pattern 4: General with people
        elif faces >= 2:
            audience['primary_demographic'].append("general consumers (25-50)")
            audience['characteristics'].extend([
                "socially engaged",
                "values community"
            ])
            audience['confidence'] = 'low'
        
        # Default fallback
        else:
            audience['primary_demographic'].append("general adult consumers (25-54)")
            audience['characteristics'].append("broad appeal")
            audience['confidence'] = 'low'
        
        # Build description
        demo = audience['primary_demographic'][0]
        chars = ", ".join(audience['characteristics'][:4])
        audience['description'] = f"Targeting {demo}. Key traits: {chars}."
        
        return audience
    
    def _infer_product_type(self, features: dict) -> dict:
        """Infer product based on what we ACTUALLY see - using visual + text signals"""
        
        product = {
            'category': 'Unknown',
            'description': '',
            'confidence': 'low'
        }
        
        # Key signals
        has_urgency = features.get('urgency_language_score', 0) > 2
        has_price = features.get('has_price', False)
        has_discount = features.get('has_discount', False)
        has_cta = features.get('has_cta', False)
        face_count = features.get('face_count', 0)
        people_score = features.get('clip_people_present_score', 0)
        
        # Pattern 1: Gym/Fitness (people working out + membership pricing)
        if face_count >= 2 and people_score > 0.6 and has_urgency and (has_price or has_discount):
            product['category'] = "Fitness/Gym Membership"
            product['description'] = "Fitness center or gym promotional membership offer with time-limited pricing. Shows people exercising with urgent call-to-action."
            product['confidence'] = 'high'
        
        # Pattern 2: Subscription service (urgency + price + CTA but no clear product)
        elif has_urgency and has_price and has_cta and face_count < 3:
            product['category'] = "Subscription Service"
            product['description'] = "Time-limited promotional offer for ongoing subscription service (streaming, software, membership, etc.)"
            product['confidence'] = 'medium'
        
        # Pattern 3: People-oriented service
        elif face_count >= 2 and people_score > 0.6:
            product['category'] = "People-Oriented Service"
            product['description'] = "Service involving people or community (fitness, wellness, social, education, etc.)"
            product['confidence'] = 'medium'
        
        # Pattern 4: E-commerce/Retail (price + discount)
        elif has_price and has_discount:
            product['category'] = "Retail/E-commerce Product"
            product['description'] = "Physical product with promotional pricing"
            product['confidence'] = 'medium'
        
        # Pattern 5: Direct response (CTA present)
        elif has_cta:
            product['category'] = "Direct Response Offer"
            product['description'] = "Advertisement with call-to-action, specific offering unclear from visuals"
            product['confidence'] = 'low'
        
        # Pattern 6: Brand awareness (no clear conversion elements)
        else:
            product['category'] = "Brand Awareness"
            product['description'] = "General brand advertisement without clear product focus or conversion elements"
            product['confidence'] = 'low'
        
        return product
    
    def _describe_emotional_appeal(self, features: dict) -> dict:
        """Describe emotions based on actual detected features"""
        
        emotions = {
            'primary_emotions': [],
            'emotional_intensity': 'medium',
            'description': ''
        }
        
        # High urgency = FOMO
        if features.get('urgency_language_score', 0) > 2:
            emotions['primary_emotions'].extend(["urgency", "FOMO (fear of missing out)", "motivation"])
            emotions['emotional_intensity'] = 'very high'
        elif features.get('urgency_language_score', 0) > 0:
            emotions['primary_emotions'].append("time-sensitivity")
            emotions['emotional_intensity'] = 'high'
        
        # Discount = value excitement
        if features.get('has_discount', False):
            emotions['primary_emotions'].extend(["excitement from savings", "smart shopping satisfaction"])
        
        # Faces = trust/connection
        if features.get('face_count', 0) > 0:
            emotions['primary_emotions'].extend(["trust", "relatability"])
        
        # Bright colors
        if features.get('avg_saturation', 128) > 140:
            emotions['primary_emotions'].append("energy")
        
        # Default
        if not emotions['primary_emotions']:
            emotions['primary_emotions'] = ["neutral interest"]
            emotions['emotional_intensity'] = 'low'
        
        emotion_list = " and ".join(emotions['primary_emotions'][:3])
        emotions['description'] = f"Designed to evoke {emotion_list}"
        
        return emotions
    
    def _describe_visual_style(self, features: dict) -> dict:
        """Describe visual style accurately"""
        
        brightness = features.get('avg_brightness', 128)
        
        return {
            'overall_feel': "energetic" if brightness > 140 else "sophisticated" if brightness < 100 else "balanced",
            'color_palette': f"{'bright' if brightness > 140 else 'dark' if brightness < 100 else 'moderate'} tones",
            'composition': "action-oriented layout" if features.get('face_count', 0) > 1 else "standard composition"
        }
    
    def _predict_engagement(self, features: dict) -> dict:
        """Predict engagement based on actual features"""
        
        # Calculate based on what we detected
        has_urgency = features.get('urgency_language_score', 0) > 0
        has_cta = features.get('has_cta', False)
        scroll_score = features.get('scrollability_score', 0.5)
        
        return {
            'scrollability': "High - urgency language grabs attention" if has_urgency else "Medium - standard visual appeal",
            'click_likelihood': "High - clear CTA with urgency" if (has_cta and has_urgency) else "Medium - some conversion elements present" if has_cta else "Low - no clear CTA",
            'overall_performance': "Good - strong urgency and CTA" if (has_urgency and has_cta) else "Average - needs optimization",
            'performance_score': round(scroll_score, 2)
        }
    
    def _identify_strengths(self, features: dict) -> list:
        """Identify actual strengths based on what we see"""
        
        strengths = []
        
        if features.get('urgency_language_score', 0) > 2:
            strengths.append("⭐ Strong urgency messaging drives immediate action")
        
        if features.get('has_cta', False):
            strengths.append("⭐ Clear call-to-action guides user behavior")
        
        if features.get('has_price', False):
            strengths.append("⭐ Transparent pricing builds trust")
        
        if features.get('has_discount', False):
            strengths.append("⭐ Promotional offer creates perceived value")
        
        if features.get('face_count', 0) > 0:
            strengths.append("⭐ Human presence builds connection and trust")
        
        if not strengths:
            strengths.append("✓ Professional presentation")
        
        return strengths[:5]
    
    def _identify_weaknesses(self, features: dict) -> list:
        """Identify actual weaknesses"""
        
        weaknesses = []
        
        if not features.get('has_cta', False):
            weaknesses.append("⚠️ Missing clear call-to-action")
        
        if features.get('scrollability_score', 0.5) < 0.4:
            weaknesses.append("⚠️ Low visual impact - may be scrolled past")
        
        if features.get('text_element_count', 0) == 0:
            weaknesses.append("⚠️ No detected text - relies entirely on visuals")
        
        if not weaknesses:
            weaknesses.append("✓ No major issues detected")
        
        return weaknesses[:5]
    
    def _generate_recommendations(self, features: dict) -> list:
        """Generate actually useful recommendations"""
        
        recs = []
        
        if not features.get('has_cta', False):
            recs.append("➤ Add clear CTA button to drive conversions")
        
        if features.get('urgency_language_score', 0) == 0:
            recs.append("➤ Add urgency language to drive immediate action")
        
        if not features.get('has_price', False):
            recs.append("➤ Consider showing pricing for transparency")
        
        if features.get('scrollability_score', 0.5) < 0.5:
            recs.append("➤ Increase visual contrast to improve scroll-stopping power")
        
        if not recs:
            recs.append("✓ Continue testing and optimizing")
        
        return recs[:5]


# Quick test function
if __name__ == "__main__":
    import sys
    gen = ImprovedAdInsightsGenerator()
    
    if len(sys.argv) > 1:
        insights = gen.generate_insights(sys.argv[1])
        print(json.dumps(insights, indent=2))