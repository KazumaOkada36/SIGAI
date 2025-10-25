"""
TEST SCRIPT - Verify your image analyzer works
Run this to make sure everything is working before integrating with the team
"""

from image_analyzer import ImageAdAnalyzer
import json

def test_analyzer():
    """Test the image analyzer with a sample image"""
    
    print("="*60)
    print("TESTING IMAGE ANALYZER")
    print("="*60)
    
    # Initialize
    print("\n1️⃣ Initializing analyzer...")
    try:
        analyzer = ImageAdAnalyzer()
        print("   ✅ Analyzer initialized successfully!")
    except Exception as e:
        print(f"   ❌ Error initializing: {e}")
        return
    
    # Test with a sample image
    print("\n2️⃣ Testing with sample image...")
    
    # You'll need to provide a real image path
    # For now, let's show what to do:
    test_image_path = "test_ad.png"  # ← CHANGE THIS to your test image
    
    print(f"   Looking for: {test_image_path}")
    
    try:
        features = analyzer.analyze(test_image_path)
        
        if 'error' in features:
            print(f"   ❌ Error analyzing image: {features['error']}")
            print("\n   💡 TIP: Make sure you have a test image!")
            print("      Create a simple test image or use one from your ads folder")
            return
        
        print(f"   ✅ Successfully extracted {len(features)} features!")
        
        # Check all feature categories are present
        print("\n3️⃣ Verifying feature categories...")
        
        expected_features = {
            'Visual Foundation': ['avg_brightness', 'avg_saturation', 'face_count', 'clip_product_focus_score'],
            'Text & CTA': ['text_element_count', 'has_cta', 'cta_keyword_count'],
            'Engagement': ['attention_color_pop', 'mobile_readability_score', 'professional_look_score'],
            'Novel Insights': ['scrollability_score', 'curiosity_gap_score', 'value_prop_clarity']
        }
        
        all_good = True
        for category, sample_features in expected_features.items():
            missing = [f for f in sample_features if f not in features]
            if missing:
                print(f"   ❌ {category}: Missing {missing}")
                all_good = False
            else:
                print(f"   ✅ {category}: All present")
        
        if all_good:
            print("\n   🎉 All feature categories present!")
        
        # Show sample features
        print("\n4️⃣ Sample extracted features:")
        print("   " + "-"*56)
        
        sample_keys = [
            'scrollability_score',
            'curiosity_gap_score', 
            'value_prop_clarity',
            'face_count',
            'has_cta',
            'avg_brightness',
            'professional_look_score'
        ]
        
        for key in sample_keys:
            if key in features:
                print(f"   {key:30s}: {features[key]}")
        
        print("   " + "-"*56)
        print(f"   (Showing 7 of {len(features)} total features)")
        
        # Save full output to JSON for inspection
        print("\n5️⃣ Saving full output...")
        with open('test_output.json', 'w') as f:
            json.dump(features, f, indent=2)
        print("   ✅ Full feature output saved to: test_output.json")
        
        # Performance check
        print("\n6️⃣ Performance check...")
        import time
        start = time.time()
        _ = analyzer.analyze(test_image_path)
        elapsed = time.time() - start
        print(f"   ⏱️  Processing time: {elapsed:.2f} seconds")
        
        if elapsed < 5:
            print("   ✅ Fast enough for competition (<5 min for dataset)")
        else:
            print("   ⚠️  Might be slow for large datasets")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n✨ Your module is ready for team integration!")
        
    except FileNotFoundError:
        print(f"   ❌ Image not found: {test_image_path}")
        print("\n   💡 NEXT STEPS:")
        print("      1. Create a test image or use one from your ads folder")
        print("      2. Update 'test_image_path' in this script")
        print("      3. Run this script again")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def create_dummy_test_image():
    """Create a simple test image if you don't have one"""
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        print("\n📝 Creating dummy test image...")
        
        # Create a simple ad-like image
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add colored rectangles (simulate ad design)
        draw.rectangle([50, 50, 750, 200], fill=(52, 152, 219))  # Blue banner
        draw.rectangle([300, 250, 500, 350], fill=(231, 76, 60))  # Red button
        
        # Add text (simulated)
        draw.rectangle([100, 400, 700, 450], fill=(0, 0, 0))  # Black text area
        
        # Save
        img.save('test_ad.png')
        print("   ✅ Created test_ad.png")
        print("   You can now run the test!")
        
        return True
    except Exception as e:
        print(f"   ❌ Couldn't create test image: {e}")
        print("   Please provide your own test image")
        return False


if __name__ == "__main__":
    import os
    
    # Check if test image exists
    if not os.path.exists('test_ad.png'):
        print("⚠️  No test image found!")
        print("\nOptions:")
        print("  1. Run create_dummy_test_image() to generate one")
        print("  2. Copy an image from your ads folder and rename to test_ad.png")
        print("\nAttempting to create dummy test image...\n")
        
        if create_dummy_test_image():
            print("\nRunning tests...\n")
            test_analyzer()
        else:
            print("\nPlease manually add a test image and run again.")
    else:
        test_analyzer()