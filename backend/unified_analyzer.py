"""
Unified Ad Analyzer - Handles BOTH images and videos
Automatically detects file type and processes accordingly
"""

import os
from pathlib import Path
from lava_extractor import LavaAdFeatureExtractor
from video_analyzer import VideoAdAnalyzer
import json

class UnifiedAdAnalyzer:
    """
    Analyzes both image and video ads
    Automatically detects file type
    """
    
    def __init__(self, model='gpt-4o-mini', max_workers=10):
        self.image_extractor = LavaAdFeatureExtractor(model=model, max_workers=max_workers)
        self.video_analyzer = VideoAdAnalyzer(model=model)
        self.model = model
    
    def is_video(self, file_path):
        """Check if file is a video"""
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
        return Path(file_path).suffix.lower() in video_extensions
    
    def is_image(self, file_path):
        """Check if file is an image"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions
    
    def process_dataset(self, input_dir, output_file='unified_features.json'):
        """
        Process all ads in directory (images AND videos)
        """
        
        print("\n" + "="*70)
        print("🎯 UNIFIED AD ANALYZER (Images + Videos)")
        print("="*70)
        print(f"\nInput directory: {input_dir}")
        print(f"Model: {self.model}")
        print("="*70 + "\n")
        
        # Find all files
        all_files = list(Path(input_dir).rglob('*'))
        
        image_files = [f for f in all_files if self.is_image(f)]
        video_files = [f for f in all_files if self.is_video(f)]
        
        print(f"📸 Found {len(image_files)} image ads")
        print(f"🎬 Found {len(video_files)} video ads")
        print(f"📊 Total: {len(image_files) + len(video_files)} ads\n")
        
        if len(image_files) + len(video_files) == 0:
            print("❌ No images or videos found!")
            return None
        
        proceed = input("▶️  Proceed with analysis? (yes/no): ")
        if proceed.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return None
        
        results = {}
        
        # Process images
        if image_files:
            print(f"\n📸 Processing {len(image_files)} images...")
            
            # Create temp directory list
            temp_dir = Path('temp_images')
            temp_dir.mkdir(exist_ok=True)
            
            # Copy to temp for batch processing
            for img in image_files:
                import shutil
                shutil.copy(img, temp_dir / img.name)
            
            # Use batch processor
            df = self.image_extractor.process_dataset(str(temp_dir), 'temp_images.json')
            
            # Load results
            with open('temp_images.json') as f:
                image_results = json.load(f)
            
            results.update(image_results)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            os.remove('temp_images.json')
            os.remove('temp_images.csv')
        
        # Process videos
        if video_files:
            print(f"\n🎬 Processing {len(video_files)} videos...")
            
            for i, video_path in enumerate(video_files, 1):
                print(f"\n[{i}/{len(video_files)}] {video_path.name}")
                
                try:
                    analysis = self.video_analyzer.analyze_video(str(video_path), num_frames='auto')
                    results[video_path.stem] = analysis
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    results[video_path.stem] = {'error': str(e)}
        
        # Save combined results
        print(f"\n💾 Saving combined results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"   Images analyzed: {len(image_files)}")
        print(f"   Videos analyzed: {len(video_files)}")
        print(f"   Total ads: {len(results)}")
        print(f"\n📁 Output: {output_file}")
        print("="*70)
        
        return results


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🎯 UNIFIED AD ANALYZER (Images + Videos)")
        print("="*70)
        print("\nUsage: python unified_analyzer.py <directory> [model]")
        print("\nExamples:")
        print("  python unified_analyzer.py ads/")
        print("  python unified_analyzer.py ads/ gpt-4o-mini")
        print("  python unified_analyzer.py ads/ gemini-pro")
        print("\nSupported formats:")
        print("  Images: .png, .jpg, .jpeg, .gif, .bmp")
        print("  Videos: .mp4, .mov, .avi, .mkv, .webm")
        print("\n" + "="*70)
        return
    
    input_dir = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else 'gpt-4o-mini'
    
    analyzer = UnifiedAdAnalyzer(model=model, max_workers=10)
    results = analyzer.process_dataset(input_dir, 'unified_features.json')


if __name__ == "__main__":
    main()