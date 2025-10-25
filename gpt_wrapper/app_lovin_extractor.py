"""
AppLovin Ad Intelligence Feature Extractor
Uses GPT-4 Vision to extract high-value signals from ad creatives

Extracts 50+ distinct features across multiple dimensions:
- Emotional/psychological signals
- Visual composition signals  
- Messaging/copy signals
- Product/content signals
- Engagement prediction signals
"""

import os
import base64
import json
from openai import OpenAI
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class AdFeatureExtractor:
    """
    Extract high-value features from ad creatives using GPT-4 Vision
    Designed for AppLovin Ad Intelligence Challenge
    """
    
    def __init__(self, api_key=None, max_workers=10):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key required!")
        
        self.client = OpenAI(api_key=api_key)
        self.max_workers = max_workers
    
    def extract_features(self, image_path):
        """
        Extract comprehensive feature set from a single ad
        
        Returns dict with 50+ features across multiple dimensions
        """
        
        # Encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # The magic prompt - designed for maximum signal extraction
        prompt = """You are an expert ad intelligence system analyzing creatives for a recommendation engine.

Extract EVERY possible signal from this advertisement that could predict engagement, clickthrough, or conversion.

**Return ONLY valid JSON in this exact structure (no markdown, no explanation):**

{
  "emotional_signals": {
    "primary_emotion": "joy/excitement/trust/fear/urgency/calm/curiosity/desire/nostalgia",
    "emotional_intensity": 1-10,
    "aspirational_appeal": 1-10,
    "humor_present": true/false,
    "creates_fomo": true/false,
    "evokes_nostalgia": true/false,
    "trust_building_elements": 1-10
  },
  "visual_composition": {
    "visual_complexity": 1-10,
    "color_scheme": "vibrant/muted/monochrome/pastel/bold/earthy",
    "dominant_colors": ["color1", "color2"],
    "contrast_level": 1-10,
    "whitespace_usage": 1-10,
    "symmetry_score": 1-10,
    "motion_energy": 1-10,
    "professional_polish": 1-10
  },
  "human_elements": {
    "people_present": true/false,
    "number_of_faces": 0-10,
    "facial_expressions": "happy/serious/surprised/neutral/varied",
    "age_demographic_shown": "children/teens/young_adults/middle_aged/seniors/mixed",
    "gender_representation": "male/female/mixed/none",
    "diversity_shown": true/false,
    "celebrity_or_influencer": true/false,
    "relatable_characters": 1-10
  },
  "text_and_messaging": {
    "text_density": 1-10,
    "headline_present": true/false,
    "headline_text": "exact text or null",
    "headline_word_count": 0-20,
    "subheading_present": true/false,
    "readability_score": 1-10,
    "power_words_count": 0-10,
    "benefit_focused": true/false,
    "problem_solution_framing": true/false
  },
  "call_to_action": {
    "cta_present": true/false,
    "cta_text": "exact text or null",
    "cta_prominence": 1-10,
    "cta_action_verb": "download/buy/learn/sign_up/try/get/join/other/none",
    "cta_urgency": 1-10,
    "cta_friction": 1-10
  },
  "product_content": {
    "product_category": "app/ecommerce/service/brand_awareness/automotive/fashion/food/finance/health/tech/gaming/other",
    "product_visible": true/false,
    "product_in_use": true/false,
    "product_benefits_shown": 1-10,
    "before_after_present": true/false,
    "price_shown": true/false,
    "discount_or_offer": true/false,
    "limited_time_offer": true/false
  },
  "branding": {
    "logo_present": true/false,
    "logo_prominence": 1-10,
    "logo_placement": "top_left/top_right/center/bottom_left/bottom_right/none",
    "brand_name_visible": true/false,
    "brand_personality": "playful/serious/luxury/friendly/innovative/trustworthy/edgy/other"
  },
  "engagement_predictors": {
    "scroll_stopping_power": 1-10,
    "first_3_sec_hook": 1-10,
    "curiosity_gap": 1-10,
    "social_proof_elements": 1-10,
    "scarcity_indicators": 1-10,
    "pattern_interruption": 1-10,
    "memability": 1-10
  },
  "technical_quality": {
    "image_quality": 1-10,
    "mobile_optimized": 1-10,
    "text_legibility": 1-10,
    "load_speed_friendly": 1-10,
    "aspect_ratio": "square/vertical/horizontal/other"
  },
  "content_type": {
    "lifestyle_imagery": true/false,
    "product_showcase": true/false,
    "testimonial_style": true/false,
    "explainer_format": true/false,
    "meme_style": true/false,
    "cinematic_style": true/false,
    "ugc_style": true/false
  },
  "psychological_triggers": {
    "authority_signals": 1-10,
    "reciprocity_elements": 1-10,
    "social_validation": 1-10,
    "commitment_consistency": 1-10,
    "liking_similarity": 1-10,
    "scarcity_urgency": 1-10
  },
  "predicted_performance": {
    "estimated_ctr": 1-10,
    "estimated_engagement": 1-10,
    "estimated_conversion_potential": 1-10,
    "virality_potential": 1-10,
    "overall_effectiveness": 1-10
  }
}

Rate ALL numeric fields on appropriate 1-10 scales. Be precise and analytical. Base everything on what you actually see in the image."""

        try:
            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON (handle markdown code blocks)
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            elif '```' in content:
                json_start = content.find('```') + 3
                json_end = content.find('```', json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # Find JSON object
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
            
            features = json.loads(json_str)
            
            # Add metadata
            features['_meta'] = {
                'ad_id': Path(image_path).stem,
                'extraction_timestamp': datetime.now().isoformat(),
                'model': 'gpt-4-vision-preview'
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
    
    def process_dataset(self, input_dir, output_file='extracted_features.json'):
        """
        Process entire dataset in parallel
        
        Args:
            input_dir: Directory containing ad images
            output_file: Where to save extracted features
        
        Returns:
            DataFrame with all extracted features
        """
        
        print("\n" + "="*70)
        print("🚀 AD INTELLIGENCE FEATURE EXTRACTOR")
        print("="*70)
        print(f"\nInput directory: {input_dir}")
        print(f"Output file: {output_file}")
        print(f"Parallel workers: {self.max_workers}")
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
        estimated_cost = len(image_files) * 0.03  # ~$0.03 per image
        
        print(f"⏱️  Estimated time: {estimated_time/60:.1f} minutes")
        print(f"💰 Estimated cost: ${estimated_cost:.2f}")
        
        proceed = input("\n▶️  Proceed? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return None
        
        # Process in parallel
        print(f"\n🔄 Processing {len(image_files)} ads in parallel...\n")
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_ad, str(img_path)): img_path
                for img_path in image_files
            }
            
            # Progress bar
            with tqdm(total=len(image_files), desc="Extracting features") as pbar:
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
        
        # Convert to DataFrame for analysis
        print("📊 Converting to DataFrame...")
        df = self.features_to_dataframe(results)
        
        # Save CSV too
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV to {csv_file}")
        
        # Print summary stats
        self.print_summary(df)
        
        return df
    
    def features_to_dataframe(self, results):
        """
        Flatten nested JSON features into DataFrame
        """
        
        rows = []
        
        for ad_id, features in results.items():
            if 'error' in features:
                continue
            
            row = {'ad_id': ad_id}
            
            # Flatten nested structure
            for category, values in features.items():
                if category == '_meta':
                    continue
                
                if isinstance(values, dict):
                    for key, value in values.items():
                        col_name = f"{category}_{key}"
                        row[col_name] = value
                else:
                    row[category] = values
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_summary(self, df):
        """Print summary statistics"""
        
        print("\n" + "="*70)
        print("📊 FEATURE EXTRACTION SUMMARY")
        print("="*70)
        
        print(f"\n📈 Dataset size: {len(df)} ads")
        print(f"📊 Total features extracted: {len(df.columns)} features")
        
        # Sample insights
        if 'engagement_predictors_scroll_stopping_power' in df.columns:
            avg_scroll_stop = df['engagement_predictors_scroll_stopping_power'].mean()
            print(f"\n🎯 Average scroll-stopping power: {avg_scroll_stop:.1f}/10")
        
        if 'predicted_performance_overall_effectiveness' in df.columns:
            avg_effectiveness = df['predicted_performance_overall_effectiveness'].mean()
            print(f"🎯 Average predicted effectiveness: {avg_effectiveness:.1f}/10")
        
        if 'call_to_action_cta_present' in df.columns:
            cta_pct = df['call_to_action_cta_present'].sum() / len(df) * 100
            print(f"🎯 Ads with CTA: {cta_pct:.1f}%")
        
        if 'human_elements_people_present' in df.columns:
            people_pct = df['human_elements_people_present'].sum() / len(df) * 100
            print(f"🎯 Ads with people: {people_pct:.1f}%")
        
        print("\n" + "="*70)
        print("✅ Feature extraction complete!")
        print("="*70)


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🎯 APPLOVIN AD INTELLIGENCE FEATURE EXTRACTOR")
        print("="*70)
        print("\nUsage: python applovin_extractor.py <ads_directory>")
        print("\nExample:")
        print("  python applovin_extractor.py images_train/image_ads/")
        print("\nRequires:")
        print("  • OPENAI_API_KEY environment variable")
        print("  • OpenAI API credits (~$0.03 per ad)")
        print("\n" + "="*70)
        return
    
    ads_dir = sys.argv[1]
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        print("\n💡 Set it with:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        return
    
    # Initialize extractor
    extractor = AdFeatureExtractor(max_workers=10)
    
    # Process dataset
    df = extractor.process_dataset(
        input_dir=ads_dir,
        output_file='applovin_features.json'
    )
    
    if df is not None:
        print(f"\n🎉 Success! Extracted features from {len(df)} ads")
        print(f"\n📁 Files created:")
        print(f"   • applovin_features.json (full data)")
        print(f"   • applovin_features.csv (tabular format)")


if __name__ == "__main__":
    main()