"""
ONE-COMMAND LOADER
Loads all 4 of your Kaggle datasets automatically
Just update the dataset IDs and run!
"""

import kagglehub
import os
from pathlib import Path
from PIL import Image
import shutil

def load_all_kaggle_datasets():
    """
    Load all 4 Kaggle ad datasets you found
    
    UPDATE THESE with your actual dataset IDs from Kaggle!
    """
    
    # ⚠️ UPDATE THESE DATASET IDs ⚠️
    # Find them on Kaggle dataset page (e.g., "username/dataset-name")
    DATASETS = {
        'internet_ads': {
            'id': 'uciml/internet-advertisements-data-set',
            'category': 'direct_response'
        },
        'car_ads': {
            'id': 'antfarol/car-sale-advertisements',  # e.g., 'username/car-sale-advertisements'
            'category': 'promotional'
        },
        'amazon_ads': {
            'id': 'sachsene/amazons-advertisements',  # e.g., 'username/amazon-advertisements'
            'category': 'ecommerce'
        },
        'social_ads': {
            'id': 'rishidamarla/social-network-advertisements',  # e.g., 'username/social-network-advertisements'
            'category': 'social'
        }
    }
    
    output_dir = "training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 LOADING ALL KAGGLE AD DATASETS")
    print("="*70)
    print("\nThis will:")
    print("  • Download datasets from Kaggle (cached after first download)")
    print("  • Extract and organize images")
    print("  • Prepare for training")
    print("="*70 + "\n")
    
    total_images = 0
    
    for name, config in DATASETS.items():
        dataset_id = config['id']
        category = config['category']
        
        # Skip if not configured
        if 'PUT_YOUR' in dataset_id:
            print(f"⏭️  Skipping {name} (dataset ID not configured)")
            continue
        
        print(f"\n📥 [{name}] Downloading: {dataset_id}")
        
        try:
            # Download with kagglehub (auto-caches!)
            dataset_path = kagglehub.dataset_download(dataset_id)
            print(f"   ✅ Downloaded to: {dataset_path}")
            
            # Organize images
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            print(f"   📂 Organizing into: {category}/")
            
            # Find all images
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
                image_files.extend(Path(dataset_path).rglob(ext))
            
            print(f"   🔍 Found {len(image_files)} image files")
            
            # Copy and process
            count = 0
            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    
                    # Convert to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save
                    output_name = f"{name}_{count:04d}.jpg"
                    output_path = os.path.join(category_dir, output_name)
                    img.save(output_path, 'JPEG', quality=95)
                    
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"   ... processed {count} images")
                    
                except Exception as e:
                    # Skip corrupted images
                    continue
            
            print(f"   ✅ Organized {count} images")
            total_images += count
            
        except Exception as e:
            print(f"   ❌ Error loading {name}: {e}")
            print(f"   💡 Check dataset ID: {dataset_id}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    if os.path.exists(output_dir):
        for category in sorted(os.listdir(output_dir)):
            category_path = os.path.join(output_dir, category)
            if os.path.isdir(category_path):
                count = len(list(Path(category_path).glob('*.jpg')))
                if count > 0:
                    print(f"  {category:25s} {count:6d} images")
    
    print("="*70)
    print(f"  {'TOTAL':25s} {total_images:6d} images")
    print("="*70)
    
    if total_images > 0:
        print(f"\n✅ SUCCESS! Data ready in: {output_dir}/")
        print("\n💡 Next steps:")
        print("   1. python train_on_kaggle.py train")
        print("   2. python hybrid_system.py your_ad.png")
    else:
        print("\n⚠️  No images loaded!")
        print("\n💡 Action needed:")
        print("   1. Find your dataset IDs on Kaggle")
        print("   2. Update DATASETS dictionary in this script")
        print("   3. Run again")


if __name__ == "__main__":
    print("\n📌 BEFORE RUNNING:")
    print("   Open this file and update the DATASET IDs with your actual Kaggle dataset names!")
    print("   Look for: 'PUT_YOUR_..._DATASET_ID_HERE'\n")
    
    proceed = input("Have you updated the dataset IDs? (yes/no): ")
    
    if proceed.lower() in ['yes', 'y']:
        load_all_kaggle_datasets()
    else:
        print("\n📝 How to find dataset IDs:")
        print("   1. Go to Kaggle dataset page")
        print("   2. Look at URL: kaggle.com/datasets/USERNAME/DATASET-NAME")
        print("   3. Dataset ID = 'USERNAME/DATASET-NAME'")
        print("\nExample:")
        print("   URL: kaggle.com/datasets/johndoe/car-advertisements")
        print("   ID:  'johndoe/car-advertisements'")
        print("\nUpdate the IDs in this script and run again!")