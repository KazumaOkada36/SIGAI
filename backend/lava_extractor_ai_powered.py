"""
AI-Powered Lava Ad Feature Extractor
Uses OpenAI to generate ALL recommendations including AB tests and roadmaps
"""

import os
import base64
import json
import requests
from pathlib import Path
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

class LavaAdFeatureExtractor:
    """
    Extract deep, actionable features using Lava gateway with AI-generated recommendations
    """
    
    def __init__(self, model='gpt-4o-mini', max_workers=10):
        """
        Initialize Lava-powered extractor
        
        Args:
            model: Which LLM to use (gpt-4o for best, gpt-4o-mini for speed)
            max_workers: Parallel processing workers
        """
        
        # Lava configuration
        self.lava_base_url = os.getenv('LAVA_BASE_URL', 'https://api.lavapayments.com/v1')
        self.lava_token = os.getenv('LAVA_FORWARD_TOKEN')
        
        if not self.lava_token:
            raise ValueError("LAVA_FORWARD_TOKEN required in .env.local!")
        
        self.model = model
        self.max_workers = max_workers
        
        print(f"✅ Initialized AI-Powered Lava extractor")
        print(f"   Model: {model}")
        print(f"   Gateway: {self.lava_base_url}")
    
    def call_lava_vision(self, image_base64, prompt):
        """
        Call vision model through Lava gateway
        
        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt for the model
            
        Returns:
            Model response
        """
        
        # Build Lava URL - routes to OpenAI
        target_url = "https://api.openai.com/v1/chat/completions"
        lava_url = f"{self.lava_base_url}/forward?u={target_url}"
        
        # Headers with Lava authentication
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.lava_token}'
        }
        
        # Request body
        request_body = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_base64}',
                                'detail': 'high'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 4000,
            'temperature': 0.3  # Balanced for creativity and consistency
        }
        
        # Make request through Lava
        response = requests.post(
            lava_url,
            headers=headers,
            json=request_body,
            timeout=90
        )
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Lava API error: {response.status_code} - {response.text}")
        
        # Get Lava request ID for tracking
        lava_request_id = response.headers.get('x-lava-request-id')
        
        # Parse response
        data = response.json()
        
        return {
            'content': data['choices'][0]['message']['content'],
            'lava_request_id': lava_request_id,
            'model_used': self.model,
            'tokens_used': data.get('usage', {})
        }
    
    def call_lava_text(self, prompt, context=None):
        """
        Call text model through Lava gateway for generating recommendations
        
        Args:
            prompt: Text prompt for the model
            context: Optional context to include
            
        Returns:
            Model response
        """
        
        # Build Lava URL - routes to OpenAI
        target_url = "https://api.openai.com/v1/chat/completions"
        lava_url = f"{self.lava_base_url}/forward?u={target_url}"
        
        # Headers with Lava authentication
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.lava_token}'
        }
        
        messages = [
            {
                'role': 'system',
                'content': 'You are an expert advertising strategist and growth marketer specializing in A/B testing and performance optimization.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
        
        # Request body
        request_body = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 3000,
            'temperature': 0.4
        }
        
        # Make request through Lava
        response = requests.post(
            lava_url,
            headers=headers,
            json=request_body,
            timeout=90
        )
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Lava API error: {response.status_code} - {response.text}")
        
        # Parse response
        data = response.json()
        
        return {
            'content': data['choices'][0]['message']['content'],
            'model_used': self.model,
            'tokens_used': data.get('usage', {})
        }
    
    def extract_features(self, image_path):
        """
        Extract comprehensive feature set with AI-generated recommendations
        """
        
        # Encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Main analysis prompt
        main_prompt = """You are an elite ad intelligence system. Analyze this advertisement and return ONLY valid JSON (no markdown, no explanation).

**CRITICAL RULES:**
1. Return ONLY the JSON object, no other text
2. Use double quotes for all strings
3. Use true/false (lowercase) for booleans
4. All numeric scores must be integers 1-10
5. Include ALL required fields
6. For improvement_roadmap and ab_test_recommendations, provide SPECIFIC, ACTIONABLE items based on what you SEE in this ad

**REQUIRED JSON STRUCTURE:**

{
  "emotional_signals": {
    "primary_emotion": "joy/excitement/trust/fear/urgency/curiosity/desire",
    "emotional_intensity": 1-10,
    "emotional_authenticity": 1-10,
    "aspirational_appeal": 1-10,
    "humor_present": true/false,
    "creates_fomo": true/false,
    "fomo_intensity": 1-10,
    "trust_building_elements": 1-10,
    "vulnerability_shown": 1-10,
    "relatability_score": 1-10
  },
  "visual_composition": {
    "visual_complexity": 1-10,
    "composition_balance": 1-10,
    "rule_of_thirds_adherence": 1-10,
    "focal_point_clarity": 1-10,
    "color_scheme": "vibrant/muted/monochrome/pastel/bold",
    "color_psychology_match": 1-10,
    "dominant_colors": ["color1", "color2", "color3"],
    "color_harmony": 1-10,
    "contrast_level": 1-10,
    "whitespace_usage": 1-10,
    "visual_hierarchy_clarity": 1-10,
    "professional_polish": 1-10,
    "production_quality": 1-10,
    "lighting_quality": 1-10
  },
  "engagement_predictors": {
    "scroll_stopping_power": 1-10,
    "first_3_sec_hook": 1-10,
    "attention_retention": 1-10,
    "curiosity_gap": 1-10,
    "social_proof_elements": 1-10,
    "scarcity_indicators": 1-10,
    "urgency_level": 1-10,
    "pattern_interruption": 1-10,
    "novelty_factor": 1-10,
    "memability": 1-10,
    "shareability": 1-10
  },
  "copy_analysis": {
    "headline_strength": 1-10,
    "headline_clarity": 1-10,
    "value_prop_clarity": 1-10,
    "benefit_focused": 1-10,
    "readability_score": 1-10,
    "power_words_used": 1-10,
    "call_to_action_present": true/false,
    "cta_strength": 1-10,
    "cta_specificity": 1-10,
    "cta_urgency": 1-10,
    "message_clarity": 1-10,
    "cta_text": "exact text of CTA if present",
    "headline_text": "exact text of headline if present",
    "body_copy_summary": "brief summary of body copy"
  },
  "predicted_performance": {
    "estimated_ctr": 1-10,
    "estimated_engagement_rate": 1-10,
    "estimated_conversion_potential": 1-10,
    "virality_potential": 1-10,
    "overall_effectiveness": 1-10
  },
  "critical_weaknesses": [
    "Specific weakness 1 with severity (HIGH/MEDIUM/LOW) - be specific about what you see",
    "Specific weakness 2 with severity - be specific about what you see",
    "Specific weakness 3 with severity - be specific about what you see"
  ],
  "key_strengths": [
    "Specific strength 1 with impact (HIGH/MEDIUM/LOW) - be specific about what you see",
    "Specific strength 2 with impact - be specific about what you see",
    "Specific strength 3 with impact - be specific about what you see"
  ],
  "executive_summary": {
    "overall_grade": "A+/A/A-/B+/B/B-/C+/C/C-/D/F",
    "one_sentence_verdict": "Concise assessment of THIS specific ad",
    "biggest_opportunity": "The #1 thing to improve in THIS ad",
    "estimated_roi_multiplier": "2x/3x/5x/10x"
  }
}

**IMPORTANT:** 
- Be extremely specific about what you see (colors, text, layout, elements)
- Base ALL assessments on THIS specific ad, not generic advice
- Extract exact text from CTA, headline, and body copy for later use"""

        try:
            print(f"🔄 Step 1: Analyzing ad features: {Path(image_path).name}")
            
            # Call through Lava for main analysis
            response = self.call_lava_vision(image_data, main_prompt)
            
            # Parse JSON from response
            content = response['content']
            json_str = self.extract_json_from_response(content)
            features = json.loads(json_str)
            
            # Validate structure
            features = self.validate_and_fix_structure(features)
            
            print(f"✅ Step 1 complete: Core features extracted")
            
            # Step 2: Generate AI-powered improvement roadmap and AB tests
            print(f"🔄 Step 2: Generating AI-powered recommendations...")
            
            recommendations = self.generate_strategic_recommendations(features, image_data)
            
            # Merge recommendations into features
            features['improvement_roadmap'] = recommendations['improvement_roadmap']
            features['ab_test_recommendations'] = recommendations['ab_test_recommendations']
            
            print(f"✅ Step 2 complete: Strategic recommendations generated")
            
            # Add metadata
            features['_meta'] = {
                'ad_id': Path(image_path).stem,
                'extraction_timestamp': datetime.now().isoformat(),
                'model': response['model_used'],
                'lava_request_id': response['lava_request_id'],
                'tokens_used': response['tokens_used'],
                'gateway': 'lava',
                'ai_generated_recommendations': True
            }
            
            print(f"✅ Successfully analyzed: {Path(image_path).name}")
            return features
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error for {image_path}: {e}")
            print(f"Response content: {content[:500]}")
            return self.create_error_response(image_path, f"JSON parsing failed: {str(e)}")
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return self.create_error_response(image_path, str(e))
    
    def generate_strategic_recommendations(self, features, image_base64):
        """
        Use AI to generate improvement roadmap and A/B test recommendations
        """
        
        # Build context from features
        context = f"""
Ad Analysis Context:
- Overall Effectiveness: {features.get('predicted_performance', {}).get('overall_effectiveness', 'N/A')}/10
- Grade: {features.get('executive_summary', {}).get('overall_grade', 'N/A')}
- Scroll-Stopping Power: {features.get('engagement_predictors', {}).get('scroll_stopping_power', 'N/A')}/10
- CTA Present: {features.get('copy_analysis', {}).get('call_to_action_present', 'N/A')}
- CTA Strength: {features.get('copy_analysis', {}).get('cta_strength', 'N/A')}/10
- CTA Text: "{features.get('copy_analysis', {}).get('cta_text', 'Not specified')}"
- Headline: "{features.get('copy_analysis', {}).get('headline_text', 'Not specified')}"
- Dominant Colors: {', '.join(features.get('visual_composition', {}).get('dominant_colors', []))}
- Visual Complexity: {features.get('visual_composition', {}).get('visual_complexity', 'N/A')}/10
- Urgency Level: {features.get('engagement_predictors', {}).get('urgency_level', 'N/A')}/10
- Social Proof: {features.get('engagement_predictors', {}).get('social_proof_elements', 'N/A')}/10

Key Weaknesses:
{chr(10).join(['- ' + w for w in features.get('critical_weaknesses', [])])}

Key Strengths:
{chr(10).join(['- ' + s for s in features.get('key_strengths', [])])}

Biggest Opportunity: {features.get('executive_summary', {}).get('biggest_opportunity', 'Not specified')}
"""
        
        prompt = f"""{context}

Based on the analysis above and by looking at the actual advertisement image, generate a comprehensive improvement strategy.

Return ONLY valid JSON with this structure:

{{
  "improvement_roadmap": {{
    "quick_wins": [
      {{
        "action": "Specific, actionable step based on what you see (e.g., 'Increase CTA button size from current 14px to 18px font and change color from #{features.get('visual_composition', {}).get('dominant_colors', ['000000'])[0]} to #FF6B35')",
        "impact": "Expected improvement with specific percentage based on industry benchmarks (e.g., '+12-15% CTR')",
        "effort": "low",
        "priority": 1
      }}
    ],
    "medium_term": [
      {{
        "action": "Specific improvement requiring 2-4 weeks",
        "impact": "Expected improvement with percentage",
        "effort": "medium",
        "priority": 2
      }}
    ],
    "long_term": [
      {{
        "action": "Strategic change for long-term impact",
        "impact": "Expected improvement with percentage",
        "effort": "high",
        "priority": 3
      }}
    ]
  }},
  "ab_test_recommendations": [
    {{
      "test": "Specific element to test (e.g., 'CTA Button: Current Blue vs High-Contrast Orange')",
      "hypothesis": "Why this change will work based on psychology and what you see in the ad",
      "expected_lift": "10-15% (based on industry benchmarks for this type of change)",
      "variant_suggestion": "Exact specification (e.g., 'Change button color from #4A90E2 to #FF6B35, increase size by 25%, add drop shadow')"
    }}
  ]]
}}

**CRITICAL REQUIREMENTS:**

For improvement_roadmap:
- Generate 4-6 quick wins (implementable in 1-7 days, low effort)
- Generate 3-4 medium term items (2-4 weeks, medium effort)
- Generate 2-3 long term items (strategic changes, high effort)
- Every action must reference specific elements you see in the ad (colors, text, layout)
- Every impact must include specific percentage ranges based on industry data
- Sort by priority (1 = highest impact/lowest effort)

For ab_test_recommendations:
- Generate 5-7 specific A/B tests covering:
  * CTA button (color, size, text, placement)
  * Headline variations
  * Image/visual changes
  * Color scheme
  * Social proof elements
  * Urgency indicators
  * Layout variations
- Each test must be based on what you actually see in the ad
- Expected lift should be based on industry benchmarks for that type of change
- Variant suggestions must be extremely specific (exact colors, sizes, text)

Return ONLY the JSON object, no other text."""

        try:
            # Call AI to generate recommendations with image context
            response = self.call_lava_vision(image_base64, prompt)
            content = response['content']
            json_str = self.extract_json_from_response(content)
            recommendations = json.loads(json_str)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Error generating AI recommendations: {e}")
            print("Using fallback structure...")
            return {
                'improvement_roadmap': {
                    'quick_wins': [
                        {
                            'action': 'AI generation failed - manual review needed',
                            'impact': 'Unable to estimate',
                            'effort': 'low',
                            'priority': 1
                        }
                    ],
                    'medium_term': [],
                    'long_term': []
                },
                'ab_test_recommendations': [
                    {
                        'test': 'AI generation failed',
                        'hypothesis': 'Manual review recommended',
                        'expected_lift': 'Unable to estimate',
                        'variant_suggestion': 'Please analyze manually'
                    }
                ]
            }
    
    def extract_json_from_response(self, content):
        """Extract JSON from various response formats"""
        # Try to find JSON in markdown code blocks
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            return content[json_start:json_end].strip()
        elif '```' in content:
            json_start = content.find('```') + 3
            json_end = content.find('```', json_start)
            return content[json_start:json_end].strip()
        else:
            # Try to find raw JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return content[json_start:json_end]
        
        # If all else fails, return the whole content
        return content
    
    def validate_and_fix_structure(self, features):
        """Validate and fix the JSON structure to ensure frontend compatibility"""
        
        # Ensure all required top-level keys exist
        required_keys = [
            'emotional_signals',
            'visual_composition',
            'engagement_predictors',
            'copy_analysis',
            'predicted_performance',
            'critical_weaknesses',
            'key_strengths',
            'executive_summary'
        ]
        
        for key in required_keys:
            if key not in features:
                features[key] = self.get_default_structure(key)
        
        # Validate numeric ranges
        features = self.validate_numeric_scores(features)
        
        # Validate lists
        if not isinstance(features.get('critical_weaknesses'), list):
            features['critical_weaknesses'] = []
        if not isinstance(features.get('key_strengths'), list):
            features['key_strengths'] = []
        
        return features
    
    def validate_numeric_scores(self, features):
        """Ensure all numeric scores are in valid range"""
        numeric_sections = [
            'emotional_signals',
            'visual_composition',
            'engagement_predictors',
            'copy_analysis',
            'predicted_performance'
        ]
        
        for section in numeric_sections:
            if section in features and isinstance(features[section], dict):
                for key, value in features[section].items():
                    if isinstance(value, (int, float)):
                        # Clamp to 1-10 range
                        features[section][key] = max(1, min(10, int(value)))
        
        return features
    
    def get_default_structure(self, key):
        """Get default structure for missing keys"""
        defaults = {
            'emotional_signals': {
                'primary_emotion': 'neutral',
                'emotional_intensity': 5,
                'emotional_authenticity': 5,
                'aspirational_appeal': 5,
                'humor_present': False,
                'creates_fomo': False,
                'fomo_intensity': 5,
                'trust_building_elements': 5,
                'vulnerability_shown': 5,
                'relatability_score': 5
            },
            'visual_composition': {
                'visual_complexity': 5,
                'composition_balance': 5,
                'rule_of_thirds_adherence': 5,
                'focal_point_clarity': 5,
                'color_scheme': 'balanced',
                'color_psychology_match': 5,
                'dominant_colors': ['unknown'],
                'color_harmony': 5,
                'contrast_level': 5,
                'whitespace_usage': 5,
                'visual_hierarchy_clarity': 5,
                'professional_polish': 5,
                'production_quality': 5,
                'lighting_quality': 5
            },
            'engagement_predictors': {
                'scroll_stopping_power': 5,
                'first_3_sec_hook': 5,
                'attention_retention': 5,
                'curiosity_gap': 5,
                'social_proof_elements': 5,
                'scarcity_indicators': 5,
                'urgency_level': 5,
                'pattern_interruption': 5,
                'novelty_factor': 5,
                'memability': 5,
                'shareability': 5
            },
            'copy_analysis': {
                'headline_strength': 5,
                'headline_clarity': 5,
                'value_prop_clarity': 5,
                'benefit_focused': 5,
                'readability_score': 5,
                'power_words_used': 5,
                'call_to_action_present': False,
                'cta_strength': 5,
                'cta_specificity': 5,
                'cta_urgency': 5,
                'message_clarity': 5
            },
            'predicted_performance': {
                'estimated_ctr': 5,
                'estimated_engagement_rate': 5,
                'estimated_conversion_potential': 5,
                'virality_potential': 5,
                'overall_effectiveness': 5
            },
            'critical_weaknesses': [
                'Unable to analyze - default response'
            ],
            'key_strengths': [
                'Unable to analyze - default response'
            ],
            'executive_summary': {
                'overall_grade': 'C',
                'one_sentence_verdict': 'Analysis incomplete',
                'biggest_opportunity': 'Unable to determine',
                'estimated_roi_multiplier': '1.5x'
            }
        }
        
        return defaults.get(key, {})
    
    def create_error_response(self, image_path, error_message):
        """Create a valid error response structure"""
        return {
            '_meta': {
                'ad_id': Path(image_path).stem,
                'error': error_message,
                'extraction_timestamp': datetime.now().isoformat()
            },
            'error': error_message
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🚀 AI-POWERED AD INTELLIGENCE EXTRACTOR")
        print("="*70)
        print("\nUsage: python lava_extractor_ai_powered.py <image_path>")
        print("\nExample:")
        print("  python lava_extractor_ai_powered.py path/to/ad.png")
        print("\nFeatures:")
        print("  ✅ AI-generated improvement roadmaps")
        print("  ✅ AI-generated A/B test recommendations")
        print("  ✅ Specific, actionable insights")
        print("  ✅ Industry benchmark-based estimates")
        print("\n" + "="*70)
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Initialize extractor
    extractor = LavaAdFeatureExtractor(model='gpt-4o-mini')
    
    # Extract features
    features = extractor.extract_features(image_path)
    
    # Save result
    output_file = f"{Path(image_path).stem}_ai_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")
    
    # Print summary
    if 'executive_summary' in features:
        print("\n" + "="*70)
        print("📊 ANALYSIS SUMMARY")
        print("="*70)
        exec_sum = features['executive_summary']
        print(f"\nGrade: {exec_sum.get('overall_grade', 'N/A')}")
        print(f"Verdict: {exec_sum.get('one_sentence_verdict', 'N/A')}")
        print(f"Biggest Opportunity: {exec_sum.get('biggest_opportunity', 'N/A')}")
        print(f"ROI Potential: {exec_sum.get('estimated_roi_multiplier', 'N/A')}")
    
    if 'improvement_roadmap' in features:
        print("\n" + "="*70)
        print("🎯 AI-GENERATED IMPROVEMENT ROADMAP")
        print("="*70)
        roadmap = features['improvement_roadmap']
        print(f"\nQuick Wins ({len(roadmap.get('quick_wins', []))} items)")
        for item in roadmap.get('quick_wins', [])[:3]:
            print(f"  • {item.get('action', 'N/A')}")
            print(f"    Impact: {item.get('impact', 'N/A')}")
    
    if 'ab_test_recommendations' in features:
        print("\n" + "="*70)
        print("🧪 AI-GENERATED A/B TEST RECOMMENDATIONS")
        print("="*70)
        for i, test in enumerate(features['ab_test_recommendations'][:3], 1):
            print(f"\nTest {i}: {test.get('test', 'N/A')}")
            print(f"  Expected Lift: {test.get('expected_lift', 'N/A')}")
            print(f"  Variant: {test.get('variant_suggestion', 'N/A')}")
    
    print("\n" + "="*70)