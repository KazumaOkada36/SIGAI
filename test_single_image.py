"""
Test Lava extractor on a single image
Quick testing tool
"""

import sys
from lava_extractor import LavaAdFeatureExtractor
import json

def test_single_image(image_path, model='gpt-4o-mini'):
    """Test extraction on a single image"""
    
    print(f"\n🔥 Testing Lava Extractor on Single Image")
    print(f"Image: {image_path}")
    print(f"Model: {model}\n")
    
    # Initialize
    extractor = LavaAdFeatureExtractor(model=model, max_workers=1)
    
    # Extract features
    print("🔄 Extracting features...")
    features = extractor.extract_features(image_path)
    
    # Display results
    if 'error' in features:
        print(f"\n❌ Error: {features['error']}")
        return
    
    print("\n✅ Extraction successful!")
    print("\n📊 Extracted Features:")
    print("="*60)
    
    # Show key features
    if 'emotional_signals' in features:
        print(f"\n😊 Emotional Signals:")
        for key, val in features['emotional_signals'].items():
            print(f"   {key}: {val}")
    
    if 'engagement_predictors' in features:
        print(f"\n🎯 Engagement Predictors:")
        for key, val in features['engagement_predictors'].items():
            print(f"   {key}: {val}")
    
    if 'predicted_performance' in features:
        print(f"\n📈 Predicted Performance:")
        for key, val in features['predicted_performance'].items():
            print(f"   {key}: {val}")
    
    # Save to file
    output_file = 'single_test_result.json'
    with open(output_file, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"\n💾 Full results saved to: {output_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_single_image.py <image_path> [model]")
        print("\nExample:")
        print("  python test_single_image.py images_train/image_ads/i0001.png")
        print("  python test_single_image.py images_train/image_ads/i0001.png gemini-pro")
    else:
        image_path = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else 'gpt-4o-mini'
        test_single_image(image_path, model)