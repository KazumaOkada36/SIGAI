"""
Direct Kaggle Dataset Loader
Uses kagglehub to automatically download and load datasets
No manual downloading needed!
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
from pathlib import Path
from PIL import Image

class KaggleAdDataLoader:
    """
    Automatically download and organize Kaggle ad datasets
    """
    
    def __init__(self, output_dir="kaggle_ads"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset identifiers (you'll need to update these with actual names)
        self.datasets = {
            'internet': 'uciml/internet-advertisements-data-set',
            # Add your other dataset identifiers here:
            # 'cars': 'username/car-sale-advertisements',
            # 'amazon': 'username/amazon-advertisements',
            # 'social': 'username/social-network-advertisements',
        }
    
    def download_all(self):
        """Download all datasets using kagglehub"""
        
        print("\n" + "="*60)
        print("📦 DOWNLOADING KAGGLE DATASETS")
        print("="*60)
        
        downloaded_paths = {}
        
        for name, dataset_id in self.datasets.items():
            print(f"\n📥 Downloading {name} ({dataset_id})...")
            
            try:
                # Download using kagglehub - it caches automatically!
                path = kagglehub.dataset_download(dataset_id)
                
                print(f"   ✅ Downloaded to: {path}")
                downloaded_paths[name] = path
                
            except Exception as e:
                print(f"   ❌ Error downloading {name}: {e}")
                print(f"   💡 Make sure dataset ID is correct: {dataset_id}")
        
        return downloaded_paths
    
    def organize_dataset(self, dataset_name, dataset_path, target_category):
        """
        Organize a downloaded dataset into training structure
        
        Args:
            dataset_name: Name of dataset (e.g., 'internet')
            dataset_path: Path where kagglehub downloaded it
            target_category: Category to organize into (e.g., 'direct_response')
        """
        
        print(f"\n📂 Organizing {dataset_name} dataset...")
        
        # Create category folder
        category_dir = os.path.join(self.output_dir, target_category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Find all images in downloaded dataset
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
            image_files.extend(Path(dataset_path).rglob(ext))
        
        print(f"   Found {len(image_files)} images")
        
        # Copy/process images
        count = 0
        for img_path in image_files:
            try:
                # Open and verify image
                img = Image.open(img_path)
                
                # Convert GIFs to static
                if img_path.suffix.lower() == '.gif':
                    img = img.convert('RGB')
                
                # Save to organized location
                output_name = f"{dataset_name}_{count:04d}.jpg"
                output_path = os.path.join(category_dir, output_name)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(output_path, 'JPEG')
                count += 1
                
            except Exception as e:
                # Skip corrupted images
                continue
        
        print(f"   ✅ Organized {count} images into {target_category}/")
        return count
    
    def load_and_organize_all(self):
        """
        Complete pipeline: Download → Organize → Ready for training
        """
        
        print("\n" + "="*60)
        print("🎯 KAGGLE AD DATA LOADER")
        print("="*60)
        print("\nThis will:")
        print("  1. Download datasets from Kaggle (auto-cached)")
        print("  2. Organize images by ad strategy")
        print("  3. Prepare for training")
        print("="*60)
        
        # Step 1: Download
        downloaded = self.download_all()
        
        if not downloaded:
            print("\n❌ No datasets downloaded. Check dataset IDs!")
            return
        
        # Step 2: Organize by strategy
        print("\n" + "="*60)
        print("📋 ORGANIZING DATASETS")
        print("="*60)
        
        total_images = 0
        
        # Map datasets to categories
        category_mapping = {
            'internet': 'direct_response',      # Banner ads → direct response
            'cars': 'promotional',              # Car ads → promotional offers
            'amazon': 'ecommerce',              # Product ads → ecommerce
            'social': 'social',                 # Social ads → social category
        }
        
        for dataset_name, dataset_path in downloaded.items():
            if dataset_name in category_mapping:
                category = category_mapping[dataset_name]
                count = self.organize_dataset(dataset_name, dataset_path, category)
                total_images += count
        
        # Step 3: Summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        for category in os.listdir(self.output_dir):
            category_path = os.path.join(self.output_dir, category)
            if os.path.isdir(category_path):
                count = len(list(Path(category_path).glob('*.jpg')))
                print(f"{category:25s} {count:5d} images")
        
        print("="*60)
        print(f"{'TOTAL':25s} {total_images:5d} images")
        print("="*60)
        print(f"\n✅ All data ready in: {self.output_dir}/")
        print(f"💡 Next step: python train_on_kaggle.py train")


def quick_load_single_dataset(dataset_id, output_category="direct_response"):
    """
    Quick function to load just one dataset
    
    Example:
        quick_load_single_dataset(
            'uciml/internet-advertisements-data-set',
            'direct_response'
        )
    """
    
    print(f"\n📥 Downloading: {dataset_id}")
    
    try:
        # Download
        path = kagglehub.dataset_download(dataset_id)
        print(f"✅ Downloaded to: {path}")
        
        # Organize
        loader = KaggleAdDataLoader(output_dir="training_data")
        count = loader.organize_dataset('dataset', path, output_category)
        
        print(f"\n✅ Loaded {count} images into training_data/{output_category}/")
        
        return path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# Convenience functions for each dataset type
def load_internet_ads():
    """Load internet advertisements dataset"""
    return quick_load_single_dataset(
        'uciml/internet-advertisements-data-set',
        'direct_response'
    )

def load_car_ads(dataset_id):
    """Load car advertisements - provide the dataset ID"""
    return quick_load_single_dataset(dataset_id, 'promotional')

def load_amazon_ads(dataset_id):
    """Load Amazon advertisements - provide the dataset ID"""
    return quick_load_single_dataset(dataset_id, 'ecommerce')

def load_social_ads(dataset_id):
    """Load social network ads - provide the dataset ID"""
    return quick_load_single_dataset(dataset_id, 'social')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load specific dataset
        dataset_id = sys.argv[1]
        category = sys.argv[2] if len(sys.argv) > 2 else 'direct_response'
        
        print(f"\n🎯 Loading dataset: {dataset_id}")
        print(f"📂 Category: {category}\n")
        
        quick_load_single_dataset(dataset_id, category)
    
    else:
        print("\n" + "="*60)
        print("📦 KAGGLE DATASET LOADER")
        print("="*60)
        print("\nUsage:")
        print("\n1. Load single dataset:")
        print("   python kaggle_loader.py <dataset_id> [category]")
        print("\n   Example:")
        print("   python kaggle_loader.py uciml/internet-advertisements-data-set direct_response")
        print("\n2. In your code:")
        print("   from kaggle_loader import quick_load_single_dataset")
        print("   quick_load_single_dataset('username/dataset-name', 'category')")
        print("\n3. Load specific types:")
        print("   from kaggle_loader import load_internet_ads")
        print("   load_internet_ads()")
        print("="*60)
        
        # Show available categories
        print("\n📂 Available categories:")
        print("   • direct_response    - Performance marketing, banner ads")
        print("   • ecommerce          - Product showcase, Amazon-style")
        print("   • promotional        - Sales, discounts, limited offers")
        print("   • social             - Social media style ads")
        print("   • brand_awareness    - Brand-focused, minimal CTA")
        print("="*60)