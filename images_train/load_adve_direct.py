"""
DIRECT LOADER for OCR Advertisement Dataset
Knows exactly where to find the ADVE folder!
"""

import kagglehub
import os
from pathlib import Path
from PIL import Image

def load_adve_folder(dataset_id='suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning'):
    """
    Load advertisements from the ADVE folder
    
    This dataset has ads in: dataset/ADVE/
    """
    
    print("\n" + "="*70)
    print("🎯 LOADING OCR ADVERTISEMENT DATASET")
    print("="*70)
    print(f"\nDataset: {dataset_id}")
    print("Target folder: ADVE/ (Advertisements)")
    print("="*70 + "\n")
    
    try:
        # Download
        print("📥 Downloading dataset...")
        dataset_path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Downloaded to: {dataset_path}")
        
        # Find the ADVE folder
        print("\n🔍 Looking for ADVE folder...")
        
        adve_folders = []
        for root, dirs, files in os.walk(dataset_path):
            for d in dirs:
                if d.upper() == 'ADVE':
                    adve_path = os.path.join(root, d)
                    adve_folders.append(adve_path)
                    print(f"   ✅ Found ADVE at: {adve_path}")
        
        if not adve_folders:
            print("   ❌ ADVE folder not found!")
            print("\n📁 Available folders:")
            for root, dirs, files in os.walk(dataset_path):
                for d in dirs:
                    print(f"      {d}/")
            return 0
        
        # Get images from ADVE folder(s)
        print("\n📸 Collecting advertisement images...")
        
        ad_images = []
        for adve_path in adve_folders:
            for root, dirs, files in os.walk(adve_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        ad_images.append(os.path.join(root, file))
        
        print(f"✅ Found {len(ad_images)} advertisement images!")
        
        if len(ad_images) == 0:
            print("❌ No images found in ADVE folder!")
            return 0
        
        # Organize into training data
        print("\n📂 Organizing into training_data/...")
        
        output_dir = "training_data"
        categories = {
            'direct_response': os.path.join(output_dir, 'direct_response'),
            'promotional': os.path.join(output_dir, 'promotional'),
            'brand_awareness': os.path.join(output_dir, 'brand_awareness')
        }
        
        for cat_path in categories.values():
            os.makedirs(cat_path, exist_ok=True)
        
        # Process and distribute images
        count = 0
        for i, img_path in enumerate(ad_images):
            try:
                # Load image
                img = Image.open(img_path)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Distribute evenly across categories
                if i % 3 == 0:
                    category = 'direct_response'
                elif i % 3 == 1:
                    category = 'promotional'
                else:
                    category = 'brand_awareness'
                
                # Save
                output_name = f"adve_{count:04d}.jpg"
                output_path = os.path.join(categories[category], output_name)
                img.save(output_path, 'JPEG', quality=95)
                
                count += 1
                
                if count % 20 == 0:
                    print(f"   ... processed {count}/{len(ad_images)} images")
                
            except Exception as e:
                print(f"   ⚠️  Skipped {os.path.basename(img_path)}: {e}")
                continue
        
        # Summary
        print("\n" + "="*70)
        print("📊 FINAL SUMMARY")
        print("="*70)
        
        for category, cat_path in categories.items():
            img_count = len(list(Path(cat_path).glob('*.jpg')))
            if img_count > 0:
                print(f"  {category:25s} {img_count:5d} images")
        
        print("="*70)
        print(f"  {'TOTAL':25s} {count:5d} images")
        print("="*70)
        
        if count > 0:
            print(f"\n✅ SUCCESS! Advertisement images ready in: {output_dir}/")
            print(f"\n💡 Next steps:")
            print("   1. python train_on_kaggle.py train")
            print("   2. python hybrid_system.py your_ad.png")
        else:
            print("\n❌ No images were processed!")
        
        return count
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import sys
    
    # Default dataset ID (update if different)
    dataset_id = 'suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning'
    
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    
    print(f"\n🚀 Loading advertisements from: {dataset_id}")
    
    count = load_adve_folder(dataset_id)
    
    if count > 0:
        print(f"\n🎉 Successfully loaded {count} advertisement images!")
        print("\n🏃 Ready to train!")
    else:
        print("\n😕 No images loaded. Check dataset structure.")