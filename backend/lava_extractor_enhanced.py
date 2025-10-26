"""
AppLovin Ad Intelligence Feature Extractor - ENHANCED WITH DEEP INSIGHTS
Routes LLM calls through Lava with advanced prompting for actionable intelligence
"""

import os
import base64
import json
import requests
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

class LavaAdFeatureExtractor:
    """
    Extract deep, actionable features using Lava gateway
    Now includes improvement roadmaps and A/B test recommendations
    """
    
    def __init__(self, model='gpt-4o', max_workers=10):
        """
        Initialize Lava-powered extractor
        
        Args:
            model: Which LLM to use (gpt-4o for best insights, gpt-4o-mini for speed)
            max_workers: Parallel processing workers
        """
        
        # Lava configuration
        self.lava_base_url = os.getenv('LAVA_BASE_URL')
        self.lava_token = os.getenv('LAVA_FORWARD_TOKEN')
        
        if not self.lava_base_url or not self.lava_token:
            raise ValueError("LAVA_BASE_URL and LAVA_FORWARD_TOKEN required in .env.local!")
        
        self.model = model
        self.max_workers = max_workers
        
        print(f"✅ Initialized Lava extractor")
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
        
        # Request body with higher token limit for detailed insights
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
            'max_tokens': 4000,  # Increased for detailed roadmaps
            'temperature': 0.3  # Slightly higher for creative insights
        }
        
        # Make request through Lava
        response = requests.post(
            lava_url,
            headers=headers,
            json=request_body,
            timeout=90  # Increased timeout for complex analysis
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
    
    def extract_features(self, image_path):
        """
        Extract comprehensive feature set with actionable insights
        """
        
        # Encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Advanced prompt for deep insights
        prompt = """You are an elite ad intelligence system used by Fortune 500 companies. Your analysis has generated over $100M in ad improvements.

Analyze this advertisement with EXTREME depth and precision. Provide insights that directly impact ROI.

**CRITICAL: Return ONLY valid JSON (no markdown, no explanation):**

{
  "emotional_signals": {
    "primary_emotion": "joy/excitement/trust/fear/urgency/calm/curiosity/desire/nostalgia/aspiration",
    "emotional_intensity": 1-10,
    "emotional_authenticity": 1-10,
    "emotional_journey": "static/builds/drops/rollercoaster",
    "aspirational_appeal": 1-10,
    "humor_present": true/false,
    "humor_type": "witty/slapstick/ironic/dark/wholesome/none",
    "creates_fomo": true/false,
    "fomo_intensity": 1-10,
    "evokes_nostalgia": true/false,
    "trust_building_elements": 1-10,
    "vulnerability_shown": 1-10,
    "relatability_score": 1-10,
    "controversy_level": 1-10
  },
  "visual_composition": {
    "visual_complexity": 1-10,
    "composition_balance": 1-10,
    "rule_of_thirds_adherence": 1-10,
    "focal_point_clarity": 1-10,
    "color_scheme": "vibrant/muted/monochrome/pastel/bold/earthy/neon/natural",
    "color_psychology_match": 1-10,
    "dominant_colors": ["color1", "color2", "color3"],
    "color_harmony": 1-10,
    "contrast_level": 1-10,
    "brightness_level": 1-10,
    "saturation_level": 1-10,
    "whitespace_usage": 1-10,
    "visual_hierarchy_clarity": 1-10,
    "symmetry_score": 1-10,
    "motion_energy": 1-10,
    "dynamic_elements": 1-10,
    "professional_polish": 1-10,
    "production_quality": 1-10,
    "lighting_quality": 1-10,
    "resolution_clarity": 1-10
  },
  "engagement_predictors": {
    "scroll_stopping_power": 1-10,
    "first_3_sec_hook": 1-10,
    "attention_retention": 1-10,
    "curiosity_gap": 1-10,
    "information_gap": 1-10,
    "social_proof_elements": 1-10,
    "authority_indicators": 1-10,
    "scarcity_indicators": 1-10,
    "urgency_level": 1-10,
    "pattern_interruption": 1-10,
    "novelty_factor": 1-10,
    "memability": 1-10,
    "shareability": 1-10,
    "comment_bait": 1-10,
    "conversation_starter": 1-10
  },
  "copy_analysis": {
    "headline_strength": 1-10,
    "headline_clarity": 1-10,
    "value_prop_clarity": 1-10,
    "benefit_focused": 1-10,
    "feature_focused": 1-10,
    "readability_score": 1-10,
    "word_count": 0-100,
    "power_words_used": 1-10,
    "emotional_words_used": 1-10,
    "jargon_level": 1-10,
    "call_to_action_present": true/false,
    "cta_strength": 1-10,
    "cta_specificity": 1-10,
    "cta_urgency": 1-10,
    "message_clarity": 1-10,
    "brand_mention_prominence": 1-10
  },
  "target_audience_signals": {
    "age_demographic": "13-17/18-24/25-34/35-44/45-54/55-64/65+/broad",
    "gender_skew": "male/female/neutral/all",
    "income_level": "budget/mid-market/premium/luxury",
    "education_level": "general/college/professional/expert",
    "lifestyle_match": 1-10,
    "pain_point_addressing": 1-10,
    "desire_addressing": 1-10,
    "cultural_relevance": 1-10,
    "trend_awareness": 1-10
  },
  "brand_elements": {
    "logo_visibility": 1-10,
    "logo_placement": "optimal/acceptable/poor/absent",
    "brand_colors_used": true/false,
    "brand_recognition": 1-10,
    "brand_trust_building": 1-10,
    "brand_differentiation": 1-10,
    "brand_personality": 1-10
  },
  "conversion_elements": {
    "value_proposition_strength": 1-10,
    "offer_clarity": 1-10,
    "risk_reversal": 1-10,
    "guarantee_present": true/false,
    "testimonial_present": true/false,
    "before_after_shown": true/false,
    "results_shown": true/false,
    "credibility_indicators": 1-10,
    "next_step_clarity": 1-10,
    "friction_reduction": 1-10
  },
  "platform_optimization": {
    "mobile_optimized": 1-10,
    "desktop_optimized": 1-10,
    "facebook_feed_fit": 1-10,
    "instagram_story_fit": 1-10,
    "tiktok_fit": 1-10,
    "youtube_preroll_fit": 1-10,
    "sound_off_effectiveness": 1-10,
    "thumbnail_quality": 1-10
  },
  "predicted_performance": {
    "estimated_ctr": 1-10,
    "estimated_engagement_rate": 1-10,
    "estimated_conversion_potential": 1-10,
    "estimated_cpac": "low/medium/high",
    "virality_potential": 1-10,
    "retention_potential": 1-10,
    "overall_effectiveness": 1-10,
    "competitive_advantage": 1-10
  },
  "critical_weaknesses": [
    "Specific weakness 1 with severity (high/medium/low)",
    "Specific weakness 2 with severity",
    "Specific weakness 3 with severity"
  ],
  "key_strengths": [
    "Specific strength 1 with impact (high/medium/low)",
    "Specific strength 2 with impact",
    "Specific strength 3 with impact"
  ],
  "improvement_roadmap": {
    "quick_wins": [
      {
        "action": "Specific action to take",
        "impact": "Expected improvement",
        "effort": "low/medium/high",
        "priority": 1-5
      }
    ],
    "medium_term": [
      {
        "action": "Specific action to take",
        "impact": "Expected improvement",
        "effort": "low/medium/high",
        "priority": 1-5
      }
    ],
    "long_term": [
      {
        "action": "Specific action to take",
        "impact": "Expected improvement",
        "effort": "low/medium/high",
        "priority": 1-5
      }
    ]
  },
  "ab_test_recommendations": [
    {
      "test": "What to test",
      "hypothesis": "Why it will work",
      "expected_lift": "5-15%",
      "variant_suggestion": "Specific change to make"
    }
  ],
  "competitive_insights": {
    "uniqueness_score": 1-10,
    "market_saturation": 1-10,
    "differentiation_opportunity": 1-10,
    "trend_alignment": 1-10
  },
  "executive_summary": {
    "overall_grade": "A+/A/A-/B+/B/B-/C+/C/C-/D/F",
    "one_sentence_verdict": "Single sentence assessment",
    "biggest_opportunity": "The #1 thing to improve",
    "estimated_roi_multiplier": "1.5x/2x/3x/5x/10x potential"
  }
}

Be brutally honest, specific, and actionable. Every score must be justified by what you see. Focus on ACTIONABLE insights that can be implemented immediately."""

        try:
            # Call through Lava
            response = self.call_lava_vision(image_data, prompt)
            
            # Parse JSON from response
            content = response['content']
            
            # Extract JSON (handle various formats)
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            elif '```' in content:
                json_start = content.find('```') + 3
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
            
            features = json.loads(json_str)
            
            # Add metadata including Lava info
            features['_meta'] = {
                'ad_id': Path(image_path).stem,
                'extraction_timestamp': datetime.now().isoformat(),
                'model': response['model_used'],
                'lava_request_id': response['lava_request_id'],
                'tokens_used': response['tokens_used'],
                'gateway': 'lava'
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return {
                '_meta': {
                    'ad_id': Path(image_path).stem,
                    'error': str(e)
                }
            }
    
    def process_single_ad(self, image_path):
        """Process single ad with error handling"""
        try:
            features = self.extract_features(image_path)
            return image_path, features
        except Exception as e:
            return image_path, {'error': str(e)}
    
    def process_dataset(self, input_dir, output_file='enhanced_ad_insights.json'):
        """
        Process entire dataset in parallel using Lava
        
        Args:
            input_dir: Directory containing ad images
            output_file: Where to save extracted features
        """
        
        print("\n" + "="*70)
        print("🚀 ENHANCED AD INTELLIGENCE EXTRACTOR - WITH ACTIONABLE INSIGHTS")
        print("="*70)
        print(f"\nInput directory: {input_dir}")
        print(f"Output file: {output_file}")
        print(f"Model: {self.model}")
        print(f"Parallel workers: {self.max_workers}")
        print("\n📊 NEW FEATURES:")
        print("   • Detailed improvement roadmaps")
        print("   • A/B test recommendations")
        print("   • Competitive insights")
        print("   • Executive summaries")
        print("="*70 + "\n")
        
        # Find all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(Path(input_dir).rglob(ext))
        
        print(f"📸 Found {len(image_files)} ad images")
        
        if len(image_files) == 0:
            print("❌ No images found!")
            return None
        
        # Estimate time and cost
        estimated_time = (len(image_files) / self.max_workers) * 5  # ~5 sec per call
        
        print(f"⏱️  Estimated time: {estimated_time/60:.1f} minutes")
        print(f"🔥 Using {self.model} for deep analysis")
        
        proceed = input("\n▶️  Proceed? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return None
        
        # Process in parallel
        print(f"\n🔄 Processing {len(image_files)} ads with deep analysis...\n")
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_ad, str(img_path)): img_path
                for img_path in image_files
            }
            
            with tqdm(total=len(image_files), desc="Extracting insights") as pbar:
                for future in as_completed(future_to_path):
                    img_path, features = future.result()
                    ad_id = Path(img_path).stem
                    results[ad_id] = features
                    pbar.update(1)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Processed {len(results)} ads in {elapsed:.1f} seconds")
        print(f"⚡ Average: {elapsed/len(results):.2f} sec/ad")
        
        # Save results
        print(f"\n💾 Saving to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Convert to DataFrame
        print("📊 Converting to DataFrame...")
        df = self.features_to_dataframe(results)
        
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV to {csv_file}")
        
        # Generate summary report
        print("\n📋 Generating summary report...")
        self.generate_summary_report(results)
        
        print("\n" + "="*70)
        print("✅ Deep feature extraction complete!")
        print("="*70)
        
        return df
    
    def features_to_dataframe(self, results):
        """Flatten nested JSON features into DataFrame"""
        rows = []
        
        for ad_id, features in results.items():
            if 'error' in features:
                continue
            
            row = {'ad_id': ad_id}
            
            for category, values in features.items():
                if category == '_meta':
                    row['lava_request_id'] = values.get('lava_request_id')
                    row['model_used'] = values.get('model')
                    continue
                
                if category in ['critical_weaknesses', 'key_strengths', 'improvement_roadmap', 'ab_test_recommendations']:
                    # Store complex objects as JSON strings
                    row[category] = json.dumps(values)
                    continue
                
                if isinstance(values, dict):
                    for key, value in values.items():
                        col_name = f"{category}_{key}"
                        if isinstance(value, (list, dict)):
                            row[col_name] = json.dumps(value)
                        else:
                            row[col_name] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_report(self, results):
        """Generate a human-readable summary report"""
        
        report_file = 'insights_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AD INTELLIGENCE SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            total_ads = len(results)
            successful = sum(1 for r in results.values() if 'error' not in r)
            
            f.write(f"Total Ads Analyzed: {total_ads}\n")
            f.write(f"Successful Analyses: {successful}\n")
            f.write(f"Failed: {total_ads - successful}\n\n")
            
            # Grade distribution
            grades = {}
            for ad_id, features in results.items():
                if 'error' in features or 'executive_summary' not in features:
                    continue
                grade = features['executive_summary'].get('overall_grade', 'N/A')
                grades[grade] = grades.get(grade, 0) + 1
            
            f.write("Grade Distribution:\n")
            for grade in sorted(grades.keys()):
                f.write(f"  {grade}: {grades[grade]} ads\n")
            f.write("\n")
            
            # Top performing ads
            f.write("="*70 + "\n")
            f.write("TOP 5 PERFORMING ADS\n")
            f.write("="*70 + "\n\n")
            
            # Sort by grade
            grade_order = {'A+': 10, 'A': 9, 'A-': 8, 'B+': 7, 'B': 6, 'B-': 5, 'C+': 4, 'C': 3, 'C-': 2, 'D': 1, 'F': 0}
            sorted_ads = sorted(
                [(ad_id, f) for ad_id, f in results.items() if 'error' not in f and 'executive_summary' in f],
                key=lambda x: grade_order.get(x[1]['executive_summary'].get('overall_grade', 'F'), 0),
                reverse=True
            )[:5]
            
            for i, (ad_id, features) in enumerate(sorted_ads, 1):
                exec_summary = features.get('executive_summary', {})
                f.write(f"{i}. {ad_id}\n")
                f.write(f"   Grade: {exec_summary.get('overall_grade', 'N/A')}\n")
                f.write(f"   Verdict: {exec_summary.get('one_sentence_verdict', 'N/A')}\n")
                f.write(f"   ROI Potential: {exec_summary.get('estimated_roi_multiplier', 'N/A')}\n\n")
            
            # Common weaknesses
            f.write("="*70 + "\n")
            f.write("COMMON WEAKNESSES ACROSS ADS\n")
            f.write("="*70 + "\n\n")
            
            all_weaknesses = []
            for ad_id, features in results.items():
                if 'error' not in features and 'critical_weaknesses' in features:
                    all_weaknesses.extend(features['critical_weaknesses'])
            
            # Count weakness patterns
            weakness_counts = {}
            for weakness in all_weaknesses:
                # Extract the main topic
                topic = weakness.split('(')[0].strip() if '(' in weakness else weakness
                weakness_counts[topic] = weakness_counts.get(topic, 0) + 1
            
            sorted_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for weakness, count in sorted_weaknesses:
                f.write(f"  • {weakness}: {count} ads\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Report saved to: " + report_file + "\n")
            f.write("="*70 + "\n")
        
        print(f"📄 Summary report saved to {report_file}")


def main():
    """Main execution with enhanced insights"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🚀 ENHANCED AD INTELLIGENCE EXTRACTOR")
        print("="*70)
        print("\nUsage: python lava_extractor.py <ads_directory> [model]")
        print("\nExamples:")
        print("  python lava_extractor.py images_train/image_ads/")
        print("  python lava_extractor.py images_train/image_ads/ gpt-4o")
        print("\nSupported models (via Lava):")
        print("  • gpt-4o (best quality - RECOMMENDED for deep insights)")
        print("  • gpt-4o-mini (fast, cheaper)")
        print("\nFeatures:")
        print("  • Detailed improvement roadmaps")
        print("  • A/B test recommendations")
        print("  • Executive summaries")
        print("  • Competitive insights")
        print("\nRequires:")
        print("  • .env.local with LAVA_BASE_URL and LAVA_FORWARD_TOKEN")
        print("\n" + "="*70)
        return
    
    ads_dir = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else 'gpt-4o'  # Default to best model
    
    # Initialize enhanced extractor
    extractor = LavaAdFeatureExtractor(model=model, max_workers=5)  # Reduced workers for quality
    
    # Process dataset
    df = extractor.process_dataset(
        input_dir=ads_dir,
        output_file='enhanced_ad_insights.json'
    )
    
    if df is not None:
        print(f"\n🎉 Success! Extracted deep insights from {len(df)} ads")
        print(f"\n📁 Files created:")
        print(f"   • enhanced_ad_insights.json (full data with roadmaps)")
        print(f"   • enhanced_ad_insights.csv (tabular format)")
        print(f"   • insights_summary.txt (executive report)")


if __name__ == "__main__":
    main()