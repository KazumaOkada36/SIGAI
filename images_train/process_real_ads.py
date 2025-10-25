"""
Process Your Real Ad Dataset
Run this on the 15 images provided by AppLovin
"""

from image_analyzer import ImageAdAnalyzer
import glob
import pandas as pd
import os
import time

def process_real_dataset(ads_folder="ads"):
    """
    Process all your real ad images from the AppLovin dataset
    
    Args:
        ads_folder: Path to folder with your ad images
    """
    
    print("="*60)
    print("PROCESSING REAL AD DATASET")
    print("="*60)
    
    # Find all image files (not videos)
    image_patterns = [
        f"{ads_folder}/*.png",
        f"{ads_folder}/*.jpg", 
        f"{ads_folder}/*.jpeg",
        f"{ads_folder}/**/*.png",
        f"{ads_folder}/**/*.jpg",
        f"{ads_folder}/**/*.jpeg"
    ]
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"\n❌ No images found in '{ads_folder}/'")
        print("\n💡 Please check:")
        print(f"   1. Does the folder '{ads_folder}/' exist?")
        print(f"   2. Are there .png or .jpg files in it?")
        print(f"   3. Try: ads_folder='path/to/your/ads'")
        return
    
    print(f"\n📁 Found {len(image_files)} image files:")
    for i, img in enumerate(sorted(image_files)[:10], 1):
        print(f"   {i}. {os.path.basename(img)}")
    if len(image_files) > 10:
        print(f"   ... and {len(image_files) - 10} more")
    
    print(f"\n🚀 Starting feature extraction...")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ImageAdAnalyzer()
    
    # Process each image
    all_features = []
    start_time = time.time()
    
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        try:
            features = analyzer.analyze(image_path)
            
            if 'error' in features:
                print(f"   ❌ Error: {features['error']}")
            else:
                print(f"   ✅ Extracted {len(features)} features")
                
                # Show some interesting features
                print(f"      • Scrollability: {features.get('scrollability_score', 'N/A')}")
                print(f"      • Faces: {features.get('face_count', 'N/A')}")
                print(f"      • Has CTA: {features.get('has_cta', 'N/A')}")
                print(f"      • Value Clarity: {features.get('value_prop_clarity', 'N/A')}")
                
                all_features.append(features)
        
        except Exception as e:
            print(f"   ❌ Error processing: {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    
    if all_features:
        # Save to CSV
        df = pd.DataFrame(all_features)
        output_file = "real_ad_features.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully processed {len(all_features)} images")
        print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/len(all_features):.2f}s per image)")
        print(f"💾 Features saved to: {output_file}")
        print(f"📊 Features extracted: {len(df.columns)} per image")
        
        # Show summary statistics
        print("\n" + "="*60)
        print("FEATURE SUMMARY")
        print("="*60)
        
        interesting_features = [
            'scrollability_score',
            'curiosity_gap_score',
            'value_prop_clarity',
            'face_count',
            'has_cta',
            'text_element_count',
            'attention_color_pop',
            'professional_look_score'
        ]
        
        print("\nKey Features Across All Images:")
        for feature in interesting_features:
            if feature in df.columns:
                if df[feature].dtype in ['int64', 'float64']:
                    avg = df[feature].mean()
                    print(f"  • {feature:30s}: avg = {avg:.2f}")
                else:
                    counts = df[feature].value_counts()
                    print(f"  • {feature:30s}: {dict(counts)}")
        
        # Show which ads are most engaging
        print("\n" + "="*60)
        print("TOP 5 MOST ENGAGING ADS (by scrollability)")
        print("="*60)
        
        if 'scrollability_score' in df.columns:
            top5 = df.nlargest(5, 'scrollability_score')[['image_path', 'scrollability_score', 'face_count', 'has_cta']]
            for idx, row in top5.iterrows():
                print(f"\n{os.path.basename(row['image_path'])}")
                print(f"  Scrollability: {row['scrollability_score']:.3f}")
                print(f"  Faces: {row['face_count']}")
                print(f"  Has CTA: {row['has_cta']}")
        
        print("\n" + "="*60)
        print("🎉 YOUR MODULE IS WORKING PERFECTLY!")
        print("="*60)
        print(f"\n📂 Open '{output_file}' to see all features")
        print("📊 Use this for your team's final submission")
        
    else:
        print("\n❌ No images were successfully processed")
        print("Please check the error messages above")


if __name__ == "__main__":
    # CHANGE THIS to match your folder structure
    ads_folder = "images_train/image_ads"  # or "images_train", "data", etc.
    
    print("\n💡 Looking for images in folder:", ads_folder)
    print("   If this is wrong, edit the 'ads_folder' variable in this script\n")
    
    # Check if common folders exist
    possible_folders = ["ads", "images_train", "data", "images", "dataset"]
    found_folders = [f for f in possible_folders if os.path.exists(f)]
    
    if found_folders and ads_folder not in found_folders:
        print(f"⚠️  Folder '{ads_folder}' not found, but I found these:")
        for folder in found_folders:
            num_images = len(glob.glob(f"{folder}/*.png") + glob.glob(f"{folder}/*.jpg"))
            print(f"   • {folder}/ ({num_images} images)")
        print(f"\n   Update 'ads_folder' variable to use one of these\n")
    
    process_real_dataset(ads_folder)