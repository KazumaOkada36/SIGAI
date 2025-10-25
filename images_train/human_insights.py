"""
Human-Readable Ad Insights Generator
Converts technical features into natural language descriptions that humans can understand
"""

from image_analyzer import ImageAdAnalyzer
import json

class AdInsightsGenerator:
    """
    Generates human-readable insights from ad features
    
    Outputs things like:
    - Target audience (e.g., "young adults, fitness enthusiasts")
    - Product/service type (e.g., "selling a mobile app")
    - Emotional appeal (e.g., "makes viewers feel excited and energetic")
    - Visual style (e.g., "minimalist, professional design")
    """
    
    def __init__(self):
        self.analyzer = ImageAdAnalyzer()
    
    def generate_insights(self, image_path: str) -> dict:
        """
        Analyze an image and generate human-readable insights
        
        Returns:
            dict with 'features' (technical) and 'insights' (human-readable)
        """
        
        # First, extract technical features
        features = self.analyzer.analyze(image_path)
        
        if 'error' in features:
            return {
                'image_path': image_path,
                'error': features['error']
            }
        
        # Generate human-readable insights
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
            'technical_features': features  # Include raw features too
        }
        
        return insights
    
    def _generate_summary(self, features: dict) -> str:
        """Generate a one-sentence summary of the ad"""
        
        parts = []
        
        # Visual style
        if features.get('clip_luxury_aesthetic_score', 0) > 0.6:
            parts.append("luxury-focused")
        elif features.get('clip_minimal_design_score', 0) > 0.6:
            parts.append("minimalist")
        elif features.get('clip_playful_tone_score', 0) > 0.6:
            parts.append("playful and energetic")
        else:
            parts.append("professional")
        
        # Ad type
        balance = features.get('brand_product_balance', 'balanced')
        if balance == 'product_focused':
            parts.append("product advertisement")
        elif balance == 'brand_focused':
            parts.append("brand awareness campaign")
        else:
            parts.append("balanced brand and product ad")
        
        # Key feature
        if features.get('face_count', 0) > 0:
            parts.append(f"featuring {int(features['face_count'])} person" + 
                        ("s" if features['face_count'] > 1 else ""))
        
        if features.get('has_cta', False):
            parts.append("with clear call-to-action")
        
        return "A " + ", ".join(parts[:3]) + "."
    
    def _infer_target_audience(self, features: dict) -> dict:
        """Infer the likely target audience with detailed psychographic profiling"""
        
        audience = {
            'primary_demographic': [],
            'characteristics': [],
            'confidence': 'medium'
        }
        
        # Get key signals
        playful = features.get('clip_playful_tone_score', 0)
        luxury = features.get('clip_luxury_aesthetic_score', 0)
        minimal = features.get('clip_minimal_design_score', 0)
        brightness = features.get('avg_brightness', 128)
        saturation = features.get('avg_saturation', 128)
        faces = features.get('face_count', 0)
        has_price = features.get('has_price', False)
        has_discount = features.get('has_discount', False)
        
        # Detailed age/demographic inference
        if playful > 0.6 and saturation > 150:
            audience['primary_demographic'].append("Gen Z and young millennials (18-29)")
            audience['characteristics'].extend(["digitally native", "values authenticity", "social media active"])
            audience['confidence'] = 'high'
        elif playful > 0.5 and brightness > 140:
            audience['primary_demographic'].append("millennials (25-40)")
            audience['characteristics'].extend(["experience-driven", "socially conscious", "tech-comfortable"])
        elif luxury > 0.6:
            if minimal > 0.5:
                audience['primary_demographic'].append("affluent professionals (35-55)")
                audience['characteristics'].extend(["high discretionary income", "quality-focused", "status-conscious"])
            else:
                audience['primary_demographic'].append("established wealthy (45-65)")
                audience['characteristics'].extend(["luxury lifestyle", "brand-loyal", "exclusivity-seeking"])
            audience['confidence'] = 'high'
        elif minimal > 0.6 and brightness < 120:
            audience['primary_demographic'].append("urban professionals (28-45)")
            audience['characteristics'].extend(["design-conscious", "minimalist lifestyle", "premium quality seekers"])
        elif has_discount or (has_price and features.get('urgency_language_score', 0) > 0):
            audience['primary_demographic'].append("value-conscious shoppers (25-54)")
            audience['characteristics'].extend(["deal-hunters", "comparison shoppers", "budget-aware"])
        else:
            audience['primary_demographic'].append("mainstream adults (25-54)")
            audience['characteristics'].append("general consumers")
        
        # Lifestyle & psychographic details
        if faces > 2:
            audience['characteristics'].extend(["family-focused", "relationship-oriented", "community-minded"])
        elif faces == 2:
            audience['characteristics'].extend(["couple/partnership demographic", "shared decision-makers"])
        elif faces == 1:
            audience['characteristics'].extend(["individual purchasers", "self-focused"])
        
        # Tech adoption level
        mobile_score = features.get('mobile_readability_score', 0)
        if mobile_score > 0.8 and playful > 0.4:
            audience['characteristics'].extend(["mobile-first", "app users", "always connected"])
        elif mobile_score > 0.7:
            audience['characteristics'].append("regular mobile users")
        
        # Purchase behavior
        if features.get('has_cta', False):
            if features.get('urgency_language_score', 0) > 2:
                audience['characteristics'].append("impulse buyers")
            else:
                audience['characteristics'].append("action-oriented")
        
        if has_price and not has_discount:
            audience['characteristics'].append("transparency-valuing")
        
        if features.get('clip_product_focus_score', 0) > 0.7:
            audience['characteristics'].append("product-research focused")
        
        # Income level hints
        if luxury > 0.5 or (minimal > 0.6 and not has_discount):
            income_level = "upper-middle to high income ($100K+)"
        elif has_discount or features.get('urgency_language_score', 0) > 2:
            income_level = "middle income, budget-conscious ($40K-$80K)"
        else:
            income_level = "broad income range ($50K-$100K)"
        
        # Create detailed description
        demo = audience['primary_demographic'][0] if audience['primary_demographic'] else "general audience"
        
        # Pick top 5 most relevant characteristics
        top_chars = audience['characteristics'][:5]
        chars_text = ", ".join(top_chars) if top_chars else "broad consumer appeal"
        
        audience['description'] = (
            f"Primary target: {demo}. "
            f"Psychographic profile: {chars_text}. "
            f"Estimated income bracket: {income_level}."
        )
        
        return audience
    
    def _infer_product_type(self, features: dict) -> dict:
        """Infer what's being advertised with detailed category analysis"""
        
        product = {
            'category': 'Unknown',
            'description': '',
            'confidence': 'low',
            'specific_examples': []
        }
        
        # Extract key signals
        product_score = features.get('clip_product_focus_score', 0)
        luxury_score = features.get('clip_luxury_aesthetic_score', 0)
        playful_score = features.get('clip_playful_tone_score', 0)
        minimal_score = features.get('clip_minimal_design_score', 0)
        people_score = features.get('clip_people_present_score', 0)
        face_count = features.get('face_count', 0)
        has_text = features.get('text_element_count', 0) > 0
        has_cta = features.get('has_cta', False)
        has_price = features.get('has_price', False)
        brightness = features.get('avg_brightness', 128)
        complexity = features.get('visual_complexity', 50)
        
        # High-confidence luxury category
        if luxury_score > 0.65:
            if product_score > 0.6:
                product['category'] = "Premium Consumer Product"
                if minimal_score > 0.5:
                    product['specific_examples'] = ["designer watches", "luxury smartphones", "premium audio equipment", "high-end fashion accessories"]
                else:
                    product['specific_examples'] = ["luxury vehicles", "jewelry", "designer handbags", "premium cosmetics"]
                product['description'] = f"High-end product targeting affluent consumers. Likely: {', '.join(product['specific_examples'][:2])}"
            else:
                product['category'] = "Luxury Service/Experience"
                product['specific_examples'] = ["5-star hotels", "private banking", "luxury travel", "premium real estate", "exclusive memberships"]
                product['description'] = f"Premium service offering. Likely: {', '.join(product['specific_examples'][:2])}"
            product['confidence'] = 'high'
        
        # High-confidence playful/youth category
        elif playful_score > 0.6 and brightness > 140:
            if has_cta and complexity > 40:
                product['category'] = "Mobile App/Game"
                product['specific_examples'] = ["mobile game", "social media app", "entertainment app", "dating app", "food delivery app"]
                product['description'] = f"Digital product for mobile users. Likely: {', '.join(product['specific_examples'][:2])}"
                product['confidence'] = 'high'
            elif face_count > 0:
                product['category'] = "Youth-Oriented Consumer Goods"
                product['specific_examples'] = ["snacks/beverages", "fast food", "energy drinks", "casual apparel", "consumer electronics"]
                product['description'] = f"Fun consumer products for younger demographics. Likely: {', '.join(product['specific_examples'][:2])}"
                product['confidence'] = 'medium'
            else:
                product['category'] = "Entertainment/Media"
                product['specific_examples'] = ["streaming service", "music platform", "gaming service", "social network"]
                product['description'] = f"Entertainment platform or service. Likely: {', '.join(product['specific_examples'][:2])}"
                product['confidence'] = 'medium'
        
        # Tech/SaaS products
        elif minimal_score > 0.6 and people_score < 0.3:
            if has_cta:
                product['category'] = "Software/SaaS Product"
                product['specific_examples'] = ["productivity software", "business tools", "cloud services", "analytics platform", "development tools"]
                product['description'] = f"B2B or prosumer software solution. Likely: {', '.join(product['specific_examples'][:2])}"
            else:
                product['category'] = "Tech Hardware"
                product['specific_examples'] = ["laptops", "tablets", "smart home devices", "wearables", "audio equipment"]
                product['description'] = f"Technology hardware product. Likely: {', '.join(product['specific_examples'][:2])}"
            product['confidence'] = 'medium'
        
        # People-focused services
        elif people_score > 0.7 or face_count > 1:
            if face_count > 2:
                product['category'] = "Family/Community Service"
                product['specific_examples'] = ["insurance (family plans)", "telecommunications", "education services", "healthcare", "family entertainment"]
                product['description'] = f"Service for families or groups. Likely: {', '.join(product['specific_examples'][:2])}"
            elif minimal_score > 0.4:
                product['category'] = "Professional Service"
                product['specific_examples'] = ["financial advisory", "career services", "consulting", "legal services", "business coaching"]
                product['description'] = f"B2B or professional service. Likely: {', '.join(product['specific_examples'][:2])}"
            else:
                product['category'] = "Personal Service"
                product['specific_examples'] = ["fitness/wellness", "healthcare", "personal finance", "dating service", "therapy/counseling"]
                product['description'] = f"Individual-focused service. Likely: {', '.join(product['specific_examples'][:2])}"
            product['confidence'] = 'medium'
        
        # Clear product showcase
        elif product_score > 0.7:
            if has_price or has_cta:
                product['category'] = "E-commerce/Retail Product"
                if luxury_score > 0.4:
                    product['specific_examples'] = ["mid-tier fashion", "home goods", "beauty products", "accessories"]
                else:
                    product['specific_examples'] = ["consumer electronics", "home appliances", "everyday goods", "sporting goods"]
                product['description'] = f"Direct-to-consumer product with clear purchase path. Likely: {', '.join(product['specific_examples'][:2])}"
                product['confidence'] = 'medium'
            else:
                product['category'] = "Physical Product"
                product['specific_examples'] = ["consumer goods", "packaged products", "electronics", "apparel"]
                product['description'] = f"Tangible product, likely retail. Possible: {', '.join(product['specific_examples'][:2])}"
                product['confidence'] = 'low'
        
        # Brand awareness (no clear product)
        else:
            product['category'] = "Brand Awareness Campaign"
            product['description'] = "General brand promotion without specific product focus. Building brand recognition and emotional connection."
            product['specific_examples'] = ["corporate branding", "brand refresh", "lifestyle branding"]
            product['confidence'] = 'medium'
        
        # Add context about pricing strategy
        if has_price and features.get('has_discount', False):
            product['description'] += " Price-competitive positioning with promotional offers."
        elif has_price:
            product['description'] += " Transparent pricing strategy."
        
        return product
    
    def _describe_emotional_appeal(self, features: dict) -> dict:
        """Describe the emotional response with psychological depth"""
        
        emotions = {
            'primary_emotions': [],
            'secondary_emotions': [],
            'emotional_intensity': 'medium',
            'description': '',
            'psychological_triggers': []
        }
        
        # Analyze emotional signals
        playful = features.get('clip_playful_tone_score', 0)
        luxury = features.get('clip_luxury_aesthetic_score', 0)
        people = features.get('clip_people_present_score', 0)
        faces = features.get('face_count', 0)
        urgency = features.get('urgency_language_score', 0)
        brightness = features.get('avg_brightness', 128)
        saturation = features.get('avg_saturation', 128)
        has_cta = features.get('has_cta', False)
        has_discount = features.get('has_discount', False)
        
        # Primary emotional drivers
        if playful > 0.65 and saturation > 150:
            emotions['primary_emotions'].extend(["excitement", "joy", "enthusiasm"])
            emotions['secondary_emotions'].extend(["spontaneity", "youthfulness"])
            emotions['psychological_triggers'].append("dopamine activation through bright, stimulating visuals")
            emotions['emotional_intensity'] = 'very high'
        elif playful > 0.5:
            emotions['primary_emotions'].extend(["happiness", "positivity"])
            emotions['secondary_emotions'].append("lightheartedness")
            emotions['emotional_intensity'] = 'high'
        
        if luxury > 0.65:
            emotions['primary_emotions'].extend(["aspiration", "desire", "sophistication"])
            emotions['secondary_emotions'].extend(["exclusivity", "status recognition"])
            emotions['psychological_triggers'].append("aspirational identity and social status signaling")
            emotions['emotional_intensity'] = 'high'
        elif luxury > 0.5:
            emotions['primary_emotions'].append("appreciation for quality")
            emotions['secondary_emotions'].append("refinement")
        
        if faces > 2:
            emotions['primary_emotions'].extend(["trust", "belonging", "connection"])
            emotions['secondary_emotions'].extend(["community", "togetherness"])
            emotions['psychological_triggers'].append("social proof and tribal belonging")
        elif faces == 2:
            emotions['primary_emotions'].extend(["partnership", "intimacy"])
            emotions['secondary_emotions'].append("shared experience")
            emotions['psychological_triggers'].append("relationship validation")
        elif faces == 1:
            emotions['primary_emotions'].extend(["trust", "personal connection"])
            emotions['secondary_emotions'].append("empathy")
            emotions['psychological_triggers'].append("face-to-face emotional resonance")
        
        if urgency > 3:
            emotions['primary_emotions'].extend(["urgency", "FOMO (fear of missing out)"])
            emotions['secondary_emotions'].extend(["scarcity anxiety", "decisiveness"])
            emotions['psychological_triggers'].append("scarcity principle and loss aversion")
            emotions['emotional_intensity'] = 'very high'
        elif urgency > 1:
            emotions['primary_emotions'].append("motivation to act")
            emotions['secondary_emotions'].append("time sensitivity")
        
        if has_discount:
            emotions['primary_emotions'].append("excitement from value")
            emotions['secondary_emotions'].extend(["smart shopping satisfaction", "winning feeling"])
            emotions['psychological_triggers'].append("reward pathway activation from perceived savings")
        
        # Visual mood indicators
        if brightness > 170 and saturation > 150:
            emotions['primary_emotions'].extend(["energy", "optimism"])
            emotions['secondary_emotions'].append("vibrancy")
            emotions['psychological_triggers'].append("high-arousal positive affect")
        elif brightness > 150:
            emotions['primary_emotions'].append("warmth")
            emotions['secondary_emotions'].append("approachability")
        elif brightness < 90:
            emotions['primary_emotions'].extend(["sophistication", "mystery"])
            emotions['secondary_emotions'].extend(["exclusivity", "intrigue"])
            emotions['psychological_triggers'].append("premium perception through restraint")
        
        # CTA emotional driver
        if has_cta:
            emotions['secondary_emotions'].append("agency and empowerment")
        
        # Remove duplicates
        emotions['primary_emotions'] = list(dict.fromkeys(emotions['primary_emotions']))
        emotions['secondary_emotions'] = list(dict.fromkeys(emotions['secondary_emotions']))
        
        # Default if no strong signals
        if not emotions['primary_emotions']:
            emotions['primary_emotions'] = ["calm", "neutral interest"]
            emotions['emotional_intensity'] = 'low'
            emotions['description'] = "Maintains a neutral, informational tone without strong emotional manipulation"
        else:
            # Create rich description
            primary_list = " and ".join(emotions['primary_emotions'][:3])
            
            if emotions['secondary_emotions']:
                secondary_list = ", ".join(emotions['secondary_emotions'][:2])
                emotions['description'] = (
                    f"Designed to evoke {primary_list}, with undertones of {secondary_list}. "
                )
            else:
                emotions['description'] = f"Designed to make viewers feel {primary_list}. "
            
            # Add psychological mechanism
            if emotions['psychological_triggers']:
                trigger = emotions['psychological_triggers'][0]
                emotions['description'] += f"Leverages {trigger}."
        
        return emotions
    
    def _describe_visual_style(self, features: dict) -> dict:
        """Describe the visual style and design approach"""
        
        style = {
            'style_descriptors': [],
            'color_palette': '',
            'composition': '',
            'overall_feel': ''
        }
        
        # Style descriptors
        if features.get('clip_luxury_aesthetic_score', 0) > 0.6:
            style['style_descriptors'].append("luxurious")
            style['style_descriptors'].append("premium")
        
        if features.get('clip_minimal_design_score', 0) > 0.6:
            style['style_descriptors'].append("minimalist")
            style['style_descriptors'].append("clean")
        
        if features.get('clip_playful_tone_score', 0) > 0.6:
            style['style_descriptors'].append("playful")
            style['style_descriptors'].append("energetic")
        
        if features.get('professional_look_score', 0) > 0.8:
            style['style_descriptors'].append("professional")
            style['style_descriptors'].append("polished")
        
        if features.get('visual_complexity', 50) < 30:
            style['style_descriptors'].append("simple")
        elif features.get('visual_complexity', 50) > 70:
            style['style_descriptors'].append("detailed")
            style['style_descriptors'].append("busy")
        
        # Color palette
        brightness = features.get('avg_brightness', 128)
        saturation = features.get('avg_saturation', 128)
        
        if brightness > 160:
            color_bright = "bright"
        elif brightness < 100:
            color_bright = "dark"
        else:
            color_bright = "moderate"
        
        if saturation > 150:
            color_sat = "vibrant"
        elif saturation < 80:
            color_sat = "muted"
        else:
            color_sat = "balanced"
        
        style['color_palette'] = f"{color_bright} with {color_sat} colors"
        
        # Composition
        if features.get('rule_of_thirds_score', 0) >= 3:
            style['composition'] = "professionally composed with strong rule-of-thirds placement"
        elif features.get('center_focus_score', 0) > 60:
            style['composition'] = "centrally focused with clear visual hierarchy"
        else:
            style['composition'] = "balanced composition"
        
        # Overall feel
        if style['style_descriptors']:
            style['overall_feel'] = ", ".join(style['style_descriptors'][:3]) + " visual style"
        else:
            style['overall_feel'] = "standard advertising aesthetic"
        
        return style
    
    def _predict_engagement(self, features: dict) -> dict:
        """Predict engagement performance"""
        
        prediction = {
            'scrollability': '',
            'click_likelihood': '',
            'overall_performance': '',
            'performance_score': 0
        }
        
        # Scrollability
        scroll_score = features.get('scrollability_score', 0.5)
        if scroll_score > 0.75:
            prediction['scrollability'] = "Very High - Users will likely stop scrolling"
        elif scroll_score > 0.6:
            prediction['scrollability'] = "High - Good attention-grabbing potential"
        elif scroll_score > 0.4:
            prediction['scrollability'] = "Moderate - May get some attention"
        else:
            prediction['scrollability'] = "Low - Likely to be scrolled past"
        
        # Click likelihood
        cta_score = features.get('cta_prominence_score', 0)
        clarity_score = features.get('value_prop_clarity', 0.5)
        
        click_score = (cta_score * 0.6 + clarity_score * 0.4)
        
        if click_score > 0.7 and features.get('has_cta', False):
            prediction['click_likelihood'] = "High - Clear CTA with compelling value proposition"
        elif click_score > 0.5:
            prediction['click_likelihood'] = "Moderate - Some clarity but could be stronger"
        else:
            prediction['click_likelihood'] = "Low - Unclear value or missing CTA"
        
        # Overall performance
        performance = (scroll_score * 0.4 + click_score * 0.3 + 
                      features.get('first_impression_score', 0.5) * 0.3)
        
        prediction['performance_score'] = round(performance, 2)
        
        if performance > 0.75:
            prediction['overall_performance'] = "Excellent - Likely to perform very well"
        elif performance > 0.6:
            prediction['overall_performance'] = "Good - Should perform above average"
        elif performance > 0.45:
            prediction['overall_performance'] = "Average - Moderate performance expected"
        else:
            prediction['overall_performance'] = "Below Average - Needs improvement"
        
        return prediction
    
    def _identify_strengths(self, features: dict) -> list:
        """Identify the ad's key strengths"""
        
        strengths = []
        
        if features.get('scrollability_score', 0) > 0.7:
            strengths.append("⭐ Highly attention-grabbing - stops users mid-scroll")
        
        if features.get('face_count', 0) > 0:
            strengths.append("⭐ Human connection through faces builds trust and relatability")
        
        if features.get('has_cta', False) and features.get('cta_prominence_score', 0) > 0.6:
            strengths.append("⭐ Clear, visible call-to-action drives user response")
        
        if features.get('value_prop_clarity', 0) > 0.8:
            strengths.append("⭐ Crystal-clear value proposition - instantly understandable")
        
        if features.get('professional_look_score', 0) > 0.8:
            strengths.append("⭐ Professional, polished design builds brand credibility")
        
        if features.get('mobile_readability_score', 0) > 0.8:
            strengths.append("⭐ Excellent mobile optimization - readable on all devices")
        
        if features.get('emotional_appeal_score', 0) > 0.7:
            strengths.append("⭐ Strong emotional appeal creates viewer connection")
        
        if features.get('first_impression_score', 0) > 0.75:
            strengths.append("⭐ Powerful first impression - memorable and striking")
        
        if not strengths:
            strengths.append("✓ Functional design with standard advertising elements")
        
        return strengths[:5]  # Top 5 strengths
    
    def _identify_weaknesses(self, features: dict) -> list:
        """Identify areas for improvement"""
        
        weaknesses = []
        
        if features.get('scrollability_score', 0) < 0.4:
            weaknesses.append("⚠️ Low scrollability - may be easily ignored in feeds")
        
        if not features.get('has_cta', False):
            weaknesses.append("⚠️ Missing call-to-action - unclear what users should do next")
        
        if features.get('face_count', 0) == 0 and features.get('clip_people_present_score', 0) < 0.3:
            weaknesses.append("⚠️ No human element - lacks personal connection")
        
        if features.get('value_prop_clarity', 0) < 0.5:
            weaknesses.append("⚠️ Unclear value proposition - message is confusing")
        
        if features.get('mobile_readability_score', 0) < 0.5:
            weaknesses.append("⚠️ Poor mobile readability - difficult to view on phones (80% of users)")
        
        if features.get('information_density', '') == 'overloaded':
            weaknesses.append("⚠️ Information overload - too cluttered and overwhelming")
        
        if features.get('information_density', '') == 'sparse':
            weaknesses.append("⚠️ Too sparse - lacks sufficient information to drive action")
        
        if features.get('text_element_count', 0) == 0:
            weaknesses.append("⚠️ No text - relies entirely on visual communication")
        
        if features.get('professional_look_score', 0) < 0.4:
            weaknesses.append("⚠️ Unprofessional appearance - may hurt brand perception")
        
        if not weaknesses:
            weaknesses.append("✓ No major weaknesses identified - solid overall execution")
        
        return weaknesses[:5]  # Top 5 weaknesses
    
    def _generate_recommendations(self, features: dict) -> list:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on weaknesses, provide specific actions
        if not features.get('has_cta', False):
            recommendations.append(
                "➤ Add a clear call-to-action button (e.g., 'Shop Now', 'Learn More', 'Download') "
                "- CTAs increase click-through rates by 200-300%"
            )
        
        if features.get('scrollability_score', 0) < 0.5:
            recommendations.append(
                "➤ Increase visual contrast and add faces to improve scroll-stopping power "
                "- consider brighter colors or central focal point"
            )
        
        if features.get('face_count', 0) == 0:
            recommendations.append(
                "➤ Consider adding human faces to create emotional connection "
                "- ads with faces see 20-40% higher engagement"
            )
        
        if features.get('mobile_readability_score', 0) < 0.7:
            recommendations.append(
                "➤ Increase text size and simplify design for mobile viewing "
                "- 80% of ad views happen on mobile devices"
            )
        
        if features.get('value_prop_clarity', 0) < 0.6:
            recommendations.append(
                "➤ Clarify your value proposition - make the benefit obvious in 2 seconds or less"
            )
        
        if features.get('information_density', '') == 'overloaded':
            recommendations.append(
                "➤ Simplify the design - remove unnecessary elements to improve comprehension"
            )
        
        if not features.get('has_price', False) and features.get('has_discount', False):
            recommendations.append(
                "➤ Consider showing pricing to increase transparency and trust"
            )
        
        if features.get('urgency_language_score', 0) == 0:
            recommendations.append(
                "➤ Add urgency language ('Limited Time', 'Today Only') to drive immediate action"
            )
        
        if not recommendations:
            recommendations.append("✓ Ad is well-optimized - continue testing and iterating")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def generate_readable_report(self, image_path: str) -> str:
        """Generate a complete human-readable report"""
        
        insights = self.generate_insights(image_path)
        
        if 'error' in insights:
            return f"Error analyzing {image_path}: {insights['error']}"
        
        report = []
        report.append("=" * 80)
        report.append(f"AD ANALYSIS REPORT: {image_path}")
        report.append("=" * 80)
        
        report.append("\n📋 SUMMARY")
        report.append("-" * 80)
        report.append(insights['summary'])
        
        report.append("\n\n👥 TARGET AUDIENCE")
        report.append("-" * 80)
        report.append(insights['target_audience']['description'])
        if insights['target_audience']['characteristics']:
            report.append("Characteristics: " + ", ".join(insights['target_audience']['characteristics']))
        
        report.append("\n\n🛍️ PRODUCT/SERVICE TYPE")
        report.append("-" * 80)
        report.append(f"Category: {insights['product_type']['category']}")
        report.append(f"Description: {insights['product_type']['description']}")
        report.append(f"Confidence: {insights['product_type']['confidence']}")
        
        report.append("\n\n💭 EMOTIONAL APPEAL")
        report.append("-" * 80)
        report.append(insights['emotional_appeal']['description'])
        report.append(f"Primary emotions: {', '.join(insights['emotional_appeal']['primary_emotions'])}")
        report.append(f"Emotional intensity: {insights['emotional_appeal']['emotional_intensity']}")
        
        report.append("\n\n🎨 VISUAL STYLE")
        report.append("-" * 80)
        report.append(f"Overall feel: {insights['visual_style']['overall_feel']}")
        report.append(f"Color palette: {insights['visual_style']['color_palette']}")
        report.append(f"Composition: {insights['visual_style']['composition']}")
        
        report.append("\n\n📈 ENGAGEMENT PREDICTION")
        report.append("-" * 80)
        report.append(f"Scrollability: {insights['engagement_prediction']['scrollability']}")
        report.append(f"Click likelihood: {insights['engagement_prediction']['click_likelihood']}")
        report.append(f"Overall performance: {insights['engagement_prediction']['overall_performance']}")
        report.append(f"Performance score: {insights['engagement_prediction']['performance_score']}/1.0")
        
        report.append("\n\n✅ STRENGTHS")
        report.append("-" * 80)
        for strength in insights['strengths']:
            report.append(strength)
        
        report.append("\n\n⚠️ AREAS FOR IMPROVEMENT")
        report.append("-" * 80)
        for weakness in insights['weaknesses']:
            report.append(weakness)
        
        report.append("\n\n💡 RECOMMENDATIONS")
        report.append("-" * 80)
        for rec in insights['recommendations']:
            report.append(rec)
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Usage example
if __name__ == "__main__":
    import sys
    
    generator = AdInsightsGenerator()
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_ad.png"  # Change to your image
    
    print(generator.generate_readable_report(image_path))