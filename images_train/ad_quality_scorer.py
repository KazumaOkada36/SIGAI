"""
Ad Quality Scoring System
Evaluates HOW GOOD an ad is, not WHAT it's selling

Focuses on:
- Ad Type (strategy being used)
- Quality Score (how well executed)
- Specific strengths/weaknesses
- Actionable improvements
"""

from image_analyzer import ImageAdAnalyzer
import json

class AdQualityScorer:
    """
    Scores ad quality based on observable features
    No guessing - just objective quality metrics
    """
    
    def __init__(self):
        self.analyzer = ImageAdAnalyzer()
    
    def score_ad(self, image_path: str) -> dict:
        """
        Score ad quality comprehensively
        
        Returns structured analysis focused on ad effectiveness
        """
        
        # Extract features
        features = self.analyzer.analyze(image_path)
        
        if 'error' in features:
            return {'error': features['error']}
        
        # Build comprehensive quality report
        report = {
            'image_path': image_path,
            'ad_type': self._classify_ad_type(features),
            'quality_scores': self._calculate_quality_scores(features),
            'overall_grade': None,  # Will calculate after scores
            'visual_analysis': self._analyze_visual_quality(features),
            'messaging_analysis': self._analyze_messaging(features),
            'conversion_elements': self._analyze_conversion_elements(features),
            'strengths': self._identify_specific_strengths(features),
            'weaknesses': self._identify_specific_weaknesses(features),
            'recommendations': self._generate_specific_recommendations(features),
            'benchmark_comparison': self._compare_to_benchmarks(features)
        }
        
        # Calculate overall grade
        report['overall_grade'] = self._calculate_overall_grade(report['quality_scores'])
        
        return report
    
    def _classify_ad_type(self, features: dict) -> dict:
        """
        Classify by MARKETING STRATEGY, not product type
        Based on observable tactics and features
        """
        
        # Extract signals
        has_urgency = features.get('urgency_language_score', 0) > 0
        has_cta = features.get('has_cta', False)
        has_price = features.get('has_price', False)
        has_discount = features.get('has_discount', False)
        text_count = features.get('text_element_count', 0)
        face_count = features.get('face_count', 0)
        
        # Classify by strategy (not product!)
        ad_types = []
        confidence = 'medium'
        primary_type = ''
        
        # 1. Direct Response / Performance Marketing
        if (has_urgency and has_cta) or (has_price and has_cta):
            ad_types.append('Direct Response')
            confidence = 'high'
            primary_type = 'Direct Response / Performance Marketing'
        
        # 2. Promotional / Limited Offer
        if has_discount or (has_urgency and has_price):
            ad_types.append('Promotional')
            if not primary_type:
                primary_type = 'Promotional / Limited-Time Offer'
                confidence = 'high'
        
        # 3. Brand Awareness
        if text_count < 3 and not has_cta and not has_urgency:
            ad_types.append('Brand Awareness')
            if not primary_type:
                primary_type = 'Brand Awareness Campaign'
                confidence = 'medium'
        
        # 4. Product Showcase
        if features.get('clip_product_focus_score', 0) > 0.7 and not has_urgency:
            ad_types.append('Product Showcase')
            if not primary_type:
                primary_type = 'Product Showcase / Education'
                confidence = 'medium'
        
        # 5. Social Proof / Testimonial
        if face_count >= 2 and text_count > 0:
            ad_types.append('Social Proof')
        
        # Default
        if not primary_type:
            primary_type = 'General Advertisement'
            confidence = 'low'
        
        return {
            'primary_type': primary_type,
            'tactics_detected': ad_types,
            'confidence': confidence,
            'description': self._describe_ad_type(primary_type, features)
        }
    
    def _describe_ad_type(self, ad_type: str, features: dict) -> str:
        """Describe what this ad type means"""
        
        descriptions = {
            'Direct Response / Performance Marketing': 
                'Designed to drive immediate action. Optimized for clicks, sign-ups, or purchases. Success measured by conversion rate.',
            
            'Promotional / Limited-Time Offer':
                'Time-sensitive promotional campaign. Creates urgency through limited availability or expiring discounts. Goal: Drive quick conversions.',
            
            'Brand Awareness Campaign':
                'Focused on building recognition and recall rather than immediate conversions. Success measured by reach and brand lift.',
            
            'Product Showcase / Education':
                'Highlights product features and benefits. Educates potential customers. Mid-funnel content for consideration phase.',
            
            'General Advertisement':
                'Multi-purpose ad with mixed objectives. Strategy unclear from creative elements.'
        }
        
        return descriptions.get(ad_type, 'Advertisement with unclear strategic intent.')
    
    def _calculate_quality_scores(self, features: dict) -> dict:
        """
        Calculate objective quality scores
        Based on ad best practices
        """
        
        scores = {}
        
        # 1. Attention Score (0-100) - Will it get noticed?
        attention_factors = []
        if features.get('face_count', 0) > 0:
            attention_factors.append(25)  # Faces grab attention
        if features.get('avg_saturation', 128) > 150:
            attention_factors.append(20)  # Bright colors
        if features.get('visual_complexity', 50) > 40:
            attention_factors.append(15)  # Visual interest
        if features.get('urgency_language_score', 0) > 0:
            attention_factors.append(20)  # Urgent language
        if features.get('has_cta', False):
            attention_factors.append(20)  # Clear action
        
        scores['attention_score'] = min(sum(attention_factors), 100)
        
        # 2. Clarity Score (0-100) - Is message clear?
        clarity_score = 50  # Start at 50
        if features.get('has_cta', False):
            clarity_score += 25  # Clear CTA
        if features.get('text_element_count', 0) > 0:
            clarity_score += 15  # Has text
        if features.get('clip_minimal_design_score', 0) > 0.5:
            clarity_score += 10  # Clean design
        
        scores['clarity_score'] = min(clarity_score, 100)
        
        # 3. Urgency Score (0-100) - Does it drive action NOW?
        urgency_words = features.get('urgency_language_score', 0)
        urgency_score = min(urgency_words * 25, 100)  # Each urgency word = +25
        if features.get('has_discount', False):
            urgency_score = min(urgency_score + 20, 100)
        
        scores['urgency_score'] = urgency_score
        
        # 4. Conversion Readiness (0-100) - Ready to convert?
        conversion_score = 0
        if features.get('has_cta', False):
            conversion_score += 40
        if features.get('has_price', False):
            conversion_score += 20
        if features.get('value_prop_clarity', 0) > 0.5:
            conversion_score += 20
        if features.get('mobile_readability_score', 0) > 0.7:
            conversion_score += 20
        
        scores['conversion_readiness'] = min(conversion_score, 100)
        
        # 5. Emotional Impact (0-100)
        emotional_score = 30  # Base
        if features.get('face_count', 0) > 0:
            emotional_score += 30
        if features.get('clip_playful_tone_score', 0) > 0.6:
            emotional_score += 20
        if features.get('clip_luxury_aesthetic_score', 0) > 0.6:
            emotional_score += 20
        
        scores['emotional_impact'] = min(emotional_score, 100)
        
        # 6. Mobile Optimization (0-100)
        mobile_score = features.get('mobile_readability_score', 0.5) * 100
        scores['mobile_optimization'] = int(mobile_score)
        
        # 7. Professional Quality (0-100)
        professional_score = 50
        if features.get('visual_complexity', 50) > 30:
            professional_score += 20
        if features.get('avg_brightness', 128) > 100 and features.get('avg_brightness', 128) < 180:
            professional_score += 15  # Good contrast
        if features.get('text_element_count', 0) > 0:
            professional_score += 15
        
        scores['professional_quality'] = min(professional_score, 100)
        
        return scores
    
    def _calculate_overall_grade(self, scores: dict) -> dict:
        """Calculate overall letter grade"""
        
        # Weight the scores
        weights = {
            'attention_score': 0.20,
            'clarity_score': 0.15,
            'urgency_score': 0.10,
            'conversion_readiness': 0.25,
            'emotional_impact': 0.10,
            'mobile_optimization': 0.10,
            'professional_quality': 0.10
        }
        
        weighted_sum = sum(scores[key] * weights[key] for key in weights)
        
        # Convert to letter grade
        if weighted_sum >= 90:
            grade = 'A'
            description = 'Excellent - High-quality ad with strong performance potential'
        elif weighted_sum >= 80:
            grade = 'B'
            description = 'Good - Solid ad with room for optimization'
        elif weighted_sum >= 70:
            grade = 'C'
            description = 'Average - Functional but needs significant improvement'
        elif weighted_sum >= 60:
            grade = 'D'
            description = 'Below Average - Major issues need addressing'
        else:
            grade = 'F'
            description = 'Poor - Requires complete redesign'
        
        return {
            'letter_grade': grade,
            'numeric_score': round(weighted_sum, 1),
            'description': description
        }
    
    def _analyze_visual_quality(self, features: dict) -> dict:
        """Analyze visual design quality"""
        
        return {
            'brightness_level': 'bright' if features.get('avg_brightness', 128) > 150 else 'dark' if features.get('avg_brightness', 128) < 100 else 'balanced',
            'color_saturation': 'vibrant' if features.get('avg_saturation', 128) > 150 else 'muted' if features.get('avg_saturation', 128) < 100 else 'moderate',
            'visual_complexity': 'high' if features.get('visual_complexity', 50) > 60 else 'low' if features.get('visual_complexity', 50) < 30 else 'moderate',
            'has_faces': features.get('face_count', 0) > 0,
            'face_count': features.get('face_count', 0),
            'assessment': self._assess_visual_quality(features)
        }
    
    def _assess_visual_quality(self, features: dict) -> str:
        """Overall visual quality assessment"""
        
        brightness = features.get('avg_brightness', 128)
        saturation = features.get('avg_saturation', 128)
        complexity = features.get('visual_complexity', 50)
        
        if saturation > 150 and brightness > 140:
            return "High-energy visual style with vibrant colors. Eye-catching and attention-grabbing."
        elif brightness < 100:
            return "Dark, sophisticated visual style. Premium aesthetic but may have lower scroll-stopping power."
        elif complexity < 30:
            return "Minimal, clean design. Professional but may lack visual interest."
        else:
            return "Balanced visual approach with moderate complexity and color usage."
    
    def _analyze_messaging(self, features: dict) -> dict:
        """Analyze text and messaging quality"""
        
        return {
            'has_text': features.get('text_element_count', 0) > 0,
            'text_elements_count': features.get('text_element_count', 0),
            'has_call_to_action': features.get('has_cta', False),
            'has_urgency': features.get('urgency_language_score', 0) > 0,
            'urgency_intensity': 'high' if features.get('urgency_language_score', 0) > 2 else 'medium' if features.get('urgency_language_score', 0) > 0 else 'none',
            'has_pricing': features.get('has_price', False),
            'has_discount': features.get('has_discount', False),
            'value_proposition_clarity': 'clear' if features.get('value_prop_clarity', 0) > 0.6 else 'unclear',
            'assessment': self._assess_messaging(features)
        }
    
    def _assess_messaging(self, features: dict) -> str:
        """Overall messaging assessment"""
        
        has_cta = features.get('has_cta', False)
        urgency = features.get('urgency_language_score', 0)
        has_price = features.get('has_price', False)
        
        if has_cta and urgency > 2 and has_price:
            return "Strong direct response messaging with clear CTA, urgency, and pricing. Optimized for immediate conversions."
        elif has_cta and urgency > 0:
            return "Action-oriented messaging with urgency elements. Good for driving conversions."
        elif has_cta:
            return "Has call-to-action but lacks urgency. May benefit from time-sensitive language."
        else:
            return "Brand-focused messaging without clear conversion elements. Better for awareness than direct response."
    
    def _analyze_conversion_elements(self, features: dict) -> dict:
        """Analyze specific conversion optimization elements"""
        
        elements = {
            'call_to_action': {
                'present': features.get('has_cta', False),
                'impact': 'High - Essential for conversions' if features.get('has_cta', False) else 'Missing - Major weakness'
            },
            'urgency_language': {
                'present': features.get('urgency_language_score', 0) > 0,
                'intensity': features.get('urgency_language_score', 0),
                'impact': 'High - Drives immediate action' if features.get('urgency_language_score', 0) > 2 else 'Medium' if features.get('urgency_language_score', 0) > 0 else 'Missing'
            },
            'pricing_transparency': {
                'present': features.get('has_price', False),
                'impact': 'Builds trust and sets expectations' if features.get('has_price', False) else 'Consider adding for transparency'
            },
            'promotional_offer': {
                'present': features.get('has_discount', False),
                'impact': 'Creates perceived value' if features.get('has_discount', False) else 'Not applicable'
            },
            'social_proof': {
                'present': features.get('face_count', 0) > 0,
                'strength': f'{features.get("face_count", 0)} person(s) shown' if features.get('face_count', 0) > 0 else 'No human element',
                'impact': 'Builds trust and connection' if features.get('face_count', 0) > 0 else 'Consider adding faces'
            },
            'mobile_optimization': {
                'score': features.get('mobile_readability_score', 0.5),
                'impact': 'Critical - Most ads viewed on mobile' if features.get('mobile_readability_score', 0.5) > 0.7 else 'Needs improvement'
            }
        }
        
        return elements
    
    def _identify_specific_strengths(self, features: dict) -> list:
        """Identify specific, measurable strengths"""
        
        strengths = []
        
        # Urgency
        urgency = features.get('urgency_language_score', 0)
        if urgency >= 3:
            strengths.append(f"⭐ Exceptional urgency messaging ({urgency} urgent elements) - Creates strong FOMO")
        elif urgency >= 2:
            strengths.append(f"⭐ Good urgency tactics ({urgency} urgent elements) - Motivates quick action")
        
        # CTA
        if features.get('has_cta', False):
            strengths.append("⭐ Clear call-to-action present - Guides user behavior effectively")
        
        # Pricing
        if features.get('has_price', False) and features.get('has_discount', False):
            strengths.append("⭐ Transparent pricing with discount - Strong value proposition")
        elif features.get('has_price', False):
            strengths.append("⭐ Pricing shown - Builds trust and sets expectations")
        
        # Visual
        if features.get('face_count', 0) >= 2:
            strengths.append(f"⭐ Multiple faces ({features.get('face_count', 0)}) - Builds strong emotional connection")
        elif features.get('face_count', 0) == 1:
            strengths.append("⭐ Human presence - Creates relatability and trust")
        
        # Attention
        if features.get('avg_saturation', 128) > 150:
            strengths.append("⭐ Vibrant colors - High scroll-stopping power")
        
        # Mobile
        if features.get('mobile_readability_score', 0) > 0.8:
            strengths.append("⭐ Excellent mobile optimization - Works well on small screens")
        
        # Scrollability
        if features.get('scrollability_score', 0.5) > 0.7:
            strengths.append("⭐ High scrollability score - Likely to stop users mid-scroll")
        
        return strengths if strengths else ["✓ Professional presentation"]
    
    def _identify_specific_weaknesses(self, features: dict) -> list:
        """Identify specific, actionable weaknesses"""
        
        weaknesses = []
        
        # Missing CTA
        if not features.get('has_cta', False):
            weaknesses.append("⚠️ CRITICAL: No clear call-to-action - Users won't know what to do next")
        
        # Low urgency
        if features.get('urgency_language_score', 0) == 0 and features.get('has_cta', False):
            weaknesses.append("⚠️ No urgency language - May lose conversions to procrastination")
        
        # Low scrollability
        if features.get('scrollability_score', 0.5) < 0.4:
            weaknesses.append("⚠️ Low scroll-stopping power - May be easily ignored in feeds")
        
        # No text detected
        if features.get('text_element_count', 0) == 0:
            weaknesses.append("⚠️ No text detected - Message may not be clear (or OCR failed)")
        
        # No faces
        if features.get('face_count', 0) == 0 and features.get('clip_people_present_score', 0) < 0.3:
            weaknesses.append("⚠️ No human element - Harder to build emotional connection")
        
        # Poor mobile
        if features.get('mobile_readability_score', 0) < 0.5:
            weaknesses.append("⚠️ Poor mobile optimization - May not work well on smartphones")
        
        # Low brightness in attention-seeking ad
        if features.get('avg_brightness', 128) < 100 and features.get('has_cta', False):
            weaknesses.append("⚠️ Dark design for conversion ad - Consider brightening for better visibility")
        
        return weaknesses if weaknesses else ["✓ No major issues detected"]
    
    def _generate_specific_recommendations(self, features: dict) -> list:
        """Generate specific, prioritized recommendations"""
        
        recs = []
        
        # Priority 1: Missing critical elements
        if not features.get('has_cta', False):
            recs.append("🔴 PRIORITY 1: Add a clear, actionable CTA button (e.g., 'Sign Up', 'Get Started', 'Shop Now')")
        
        # Priority 2: Add urgency if converting ad
        if features.get('has_cta', False) and features.get('urgency_language_score', 0) == 0:
            recs.append("🟠 PRIORITY 2: Add time-sensitive language (e.g., 'Limited Time', 'Ends Soon', 'Today Only')")
        
        # Priority 3: Improve scroll-stopping
        if features.get('scrollability_score', 0.5) < 0.5:
            if features.get('face_count', 0) == 0:
                recs.append("🟡 PRIORITY 3: Add human faces to improve scroll-stopping power and emotional connection")
            else:
                recs.append("🟡 PRIORITY 3: Increase visual contrast or use brighter colors to grab attention")
        
        # Priority 4: Mobile optimization
        if features.get('mobile_readability_score', 0) < 0.6:
            recs.append("🟡 PRIORITY 4: Optimize for mobile - ensure text is readable and CTA is easily tappable")
        
        # Priority 5: Value proposition
        if not features.get('has_price', False) and features.get('has_cta', False):
            recs.append("🔵 PRIORITY 5: Consider showing pricing or clear value proposition ('What's in it for me?')")
        
        # Priority 6: Specific tactical improvements
        if features.get('text_element_count', 0) == 0:
            recs.append("🔵 Add clear headline text to communicate value in 2 seconds or less")
        
        return recs if recs else ["✓ Ad is well-optimized - focus on A/B testing variations"]
    
    def _compare_to_benchmarks(self, features: dict) -> dict:
        """Compare to industry benchmarks"""
        
        # These are rough benchmarks based on ad best practices
        return {
            'cta_presence': {
                'your_ad': 'Yes' if features.get('has_cta', False) else 'No',
                'industry_standard': '85% of high-performing ads',
                'assessment': 'Meets standard' if features.get('has_cta', False) else 'Below standard'
            },
            'urgency_usage': {
                'your_ad': features.get('urgency_language_score', 0),
                'industry_standard': '2-3 urgency elements',
                'assessment': 'Excellent' if features.get('urgency_language_score', 0) >= 2 else 'Below average'
            },
            'human_element': {
                'your_ad': f'{features.get("face_count", 0)} face(s)',
                'industry_standard': 'At least 1 face',
                'assessment': 'Meets standard' if features.get('face_count', 0) > 0 else 'Consider adding'
            },
            'mobile_optimization': {
                'your_ad': f'{int(features.get("mobile_readability_score", 0) * 100)}%',
                'industry_standard': '75%+',
                'assessment': 'Good' if features.get('mobile_readability_score', 0) > 0.75 else 'Needs improvement'
            }
        }


# Quick test
if __name__ == "__main__":
    import sys
    
    scorer = AdQualityScorer()
    
    if len(sys.argv) > 1:
        result = scorer.score_ad(sys.argv[1])
        print(json.dumps(result, indent=2))