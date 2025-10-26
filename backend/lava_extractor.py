"""
AppLovin Ad Intelligence Feature Extractor - WITH LAVA
Routes LLM calls through Lava for multi-model support
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
    Extract features using Lava gateway
    Can use OpenAI, Gemini, Claude, or any LLM through unified API
    """
    
    def __init__(self, model='gpt-4o-mini', max_workers=10):
        """
        Initialize Lava-powered extractor
        
        Args:
            model: Which LLM to use (gpt-4o-mini, gemini-pro, claude-3-opus, etc.)
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
        
        # Build Lava URL - routes to OpenAI (or other provider)
        target_url = "https://api.openai.com/v1/chat/completions"
        lava_url = f"{self.lava_base_url}/forward?u={target_url}"
        
        # Headers with Lava authentication
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.lava_token}'
        }
        
        # Request body (OpenAI format works for most models via Lava)
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
            'max_tokens': 2000,
            'temperature': 0.1
        }
        
        # Make request through Lava
        response = requests.post(
            lava_url,
            headers=headers,
            json=request_body,
            timeout=60
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
            'model_used': self.model
        }
    
    def extract_features(self, image_path):
        """
        Extract comprehensive feature set from a single ad
        Uses Lava to route to best available LLM
        """
        
        # Encode image
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # The feature extraction prompt (same as before)
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
  "engagement_predictors": {
    "scroll_stopping_power": 1-10,
    "first_3_sec_hook": 1-10,
    "curiosity_gap": 1-10,
    "social_proof_elements": 1-10,
    "scarcity_indicators": 1-10,
    "pattern_interruption": 1-10,
    "memability": 1-10
  },
  "predicted_performance": {
    "estimated_ctr": 1-10,
    "estimated_engagement": 1-10,
    "estimated_conversion_potential": 1-10,
    "virality_potential": 1-10,
    "overall_effectiveness": 1-10
  }
}

Rate ALL numeric fields on appropriate 1-10 scales. Be precise and analytical."""

        try:
            # Call through Lava
            response = self.call_lava_vision(image_data, prompt)
            
            # Parse JSON from response
            content = response['content']
            
            # Extract JSON
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
    
    def process_dataset(self, input_dir, output_file='lava_extracted_features.json'):
        """
        Process entire dataset in parallel using Lava
        
        Args:
            input_dir: Directory containing ad images
            output_file: Where to save extracted features
        """
        
        print("\n" + "="*70)
        print("🔥 LAVA-POWERED AD FEATURE EXTRACTOR")
        print("="*70)
        print(f"\nInput directory: {input_dir}")
        print(f"Output file: {output_file}")
        print(f"Model: {self.model}")
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
        estimated_time = (len(image_files) / self.max_workers) * 3  # ~3 sec per call with Lava
        
        print(f"⏱️  Estimated time: {estimated_time/60:.1f} minutes")
        print(f"🔥 Routing through Lava gateway")
        
        proceed = input("\n▶️  Proceed? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return None
        
        # Process in parallel
        print(f"\n🔄 Processing {len(image_files)} ads in parallel...\n")
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_ad, str(img_path)): img_path
                for img_path in image_files
            }
            
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
        
        # Convert to DataFrame
        print("📊 Converting to DataFrame...")
        df = self.features_to_dataframe(results)
        
        csv_file = output_file.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV to {csv_file}")
        
        print("\n" + "="*70)
        print("✅ Feature extraction complete!")
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
                    # Include Lava metadata
                    row['lava_request_id'] = values.get('lava_request_id')
                    row['model_used'] = values.get('model')
                    continue
                
                if isinstance(values, dict):
                    for key, value in values.items():
                        col_name = f"{category}_{key}"
                        row[col_name] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Main execution with Lava"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🔥 LAVA-POWERED AD INTELLIGENCE EXTRACTOR")
        print("="*70)
        print("\nUsage: python lava_extractor.py <ads_directory> [model]")
        print("\nExamples:")
        print("  python lava_extractor.py images_train/image_ads/")
        print("  python lava_extractor.py images_train/image_ads/ gpt-4o-mini")
        print("  python lava_extractor.py images_train/image_ads/ gemini-pro")
        print("\nSupported models (via Lava):")
        print("  • gpt-4o-mini (fast, cheap)")
        print("  • gpt-4o (best quality)")
        print("  • gemini-pro (Google)")
        print("  • claude-3-opus (Anthropic)")
        print("\nRequires:")
        print("  • .env.local with LAVA_BASE_URL and LAVA_FORWARD_TOKEN")
        print("\n" + "="*70)
        return
    
    ads_dir = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else 'gpt-4o-mini'
    
    # Initialize Lava extractor
    extractor = LavaAdFeatureExtractor(model=model, max_workers=10)
    
    # Process dataset
    df = extractor.process_dataset(
        input_dir=ads_dir,
        output_file='lava_features.json'
    )
    
    if df is not None:
        print(f"\n🎉 Success! Extracted features from {len(df)} ads")
        print(f"\n📁 Files created:")
        print(f"   • lava_features.json (full data)")
        print(f"   • lava_features.csv (tabular format)")


if __name__ == "__main__":
    main()