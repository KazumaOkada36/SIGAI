"""
Load Advertisement Images from OCR Dataset
Extracts just the advertisement category from the multi-category dataset
"""

import kagglehub
import os
from pathlib import Path
from PIL import Image
import shutil

def load_ocr_advertisements(output_dir="training_data"):
    """
    Download OCR dataset and extract just the advertisement images
    """
    
    print("\n" + "="*70)
    print("📥 LOADING OCR ADVERTISEMENT DATASET")
    print("="*70)
    print("\nDataset: Scanned Images for OCR and VLM finetuning")
    print("Category: Advertisements (promotional materials)")
    print("="*70 + "\n")
    
    # Dataset ID - update this with the actual dataset ID from Kaggle
    # Look at the URL: kaggle.com/datasets/USERNAME/DATASET-NAME
    dataset_id = "suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning"  # e.g., "username/scanned-images-ocr-vlm"
    
    if "PUT_ACTUAL" in dataset_id:
        print("⚠️  Please update the dataset_id in the script!")
        print("\n💡 Find it on the Kaggle page URL:")
        print("   Example: kaggle.com/datasets/johndoe/scanned-images")
        print("   Dataset ID: 'johndoe/scanned-images'")
        return None
    
    try:
        # Download dataset
        print("📥 Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Downloaded to: {dataset_path}")
        
        # Find advertisement images
        print("\n🔍 Searching for advertisement images...")
        
        # Common folder names for advertisements in the dataset
        ad_folders = [
            'advertisements',
            'Advertisements', 
            'ads',
            'Ads',
            'promotional',
            'Promotional'
        ]
        
        ad_images = []
        
        # Search for advertisement folder
        for root, dirs, files in os.walk(dataset_path):
            # Check if this is an advertisement folder
            folder_name = os.path.basename(root).lower()
            
            if any(ad_name.lower() in folder_name for ad_name in ad_folders):
                print(f"   📂 Found advertisement folder: {root}")
                
                # Get all images from this folder
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        ad_images.append(os.path.join(root, file))
        
        if not ad_images:
            print("   ⚠️  No advertisement folder found by name, searching all images...")
            # Fallback: look for images with 'ad' in filename
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if 'ad' in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ad_images.append(os.path.join(root, file))
        
        print(f"\n✅ Found {len(ad_images)} advertisement images!")
        
        if len(ad_images) == 0:
            print("\n⚠️  No advertisement images found!")
            print("\n💡 The dataset might be organized differently.")
            print("   Listing all folders in dataset:")
            for root, dirs, files in os.walk(dataset_path):
                if dirs:
                    print(f"   📁 {root}")
                    for d in dirs:
                        print(f"      └── {d}/")
            return None
        
        # Organize into training structure
        print(f"\n📂 Organizing images into {output_dir}/...")
        
        # Create output folders
        categories = {
            'direct_response': output_dir + '/direct_response',
            'promotional': output_dir + '/promotional',
            'brand_awareness': output_dir + '/brand_awareness'
        }
        
        for cat_path in categories.values():
            os.makedirs(cat_path, exist_ok=True)
        
        # Process images (distribute across categories)
        count = 0
        for i, img_path in enumerate(ad_images):
            try:
                # Open and verify image
                img = Image.open(img_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Distribute across categories (simple rotation)
                if i % 3 == 0:
                    category = 'direct_response'
                elif i % 3 == 1:
                    category = 'promotional'
                else:
                    category = 'brand_awareness'
                
                # Save
                output_name = f"ocr_ad_{count:04d}.jpg"
                output_path = os.path.join(categories[category], output_name)
                img.save(output_path, 'JPEG', quality=95)
                
                count += 1
                
                if count % 50 == 0:
                    print(f"   ... processed {count} images")
                
            except Exception as e:
                print(f"   ⚠️  Skipped {os.path.basename(img_path)}: {e}")
                continue
        
        # Summary
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        
        for category, cat_path in categories.items():
            img_count = len(list(Path(cat_path).glob('*.jpg')))
            if img_count > 0:
                print(f"  {category:25s} {img_count:5d} images")
        
        print("="*70)
        print(f"  {'TOTAL':25s} {count:5d} images")
        print("="*70)
        
        print(f"\n✅ SUCCESS! Advertisement images ready in: {output_dir}/")
        print(f"\n💡 Next step: python train_on_kaggle.py train")
        
        return count
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def quick_load_ocr_ads(dataset_id):
    """
    Quick function to load OCR advertisement dataset
    
    Usage:
        quick_load_ocr_ads('username/scanned-images-ocr-vlm')
    """
    
    print(f"\n🎯 Loading OCR Advertisement Dataset")
    print(f"📦 Dataset: {dataset_id}\n")
    
    try:
        # Download
        path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Downloaded to: {path}")
        
        # Find ads
        print("\n🔍 Searching for advertisements...")
        
        ad_images = []
        for root, dirs, files in os.walk(path):
            folder = os.path.basename(root).lower()
            if 'ad' in folder or 'promo' in folder:
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ad_images.append(os.path.join(root, file))
        
        print(f"✅ Found {len(ad_images)} advertisement images")
        
        # Organize
        output_dir = "training_data/direct_response"
        os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        for i, img_path in enumerate(ad_images):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                output_path = f"{output_dir}/ocr_ad_{i:04d}.jpg"
                img.save(output_path, 'JPEG')
                count += 1
            except:
                continue
        
        print(f"\n✅ Organized {count} images into training_data/direct_response/")
        return count
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
        quick_load_ocr_ads(dataset_id)
    else:
        print("\n" + "="*70)
        print("📦 OCR ADVERTISEMENT DATASET LOADER")
        print("="*70)
        print("\nUsage:")
        print("  python load_ocr_ads.py <dataset_id>")
        print("\nExample:")
        print("  python load_ocr_ads.py username/scanned-images-ocr-vlm")
        print("\n💡 Find dataset ID on Kaggle page URL")
        print("="*70)
        
        # Try to load anyway
        load_ocr_advertisements()