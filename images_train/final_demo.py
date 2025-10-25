"""
FINAL DEMO - Quality-Focused Ad Analysis
No guessing products - just tells you HOW GOOD the ad is
"""

from quality_pdf_generator import QualityPDFGenerator
import os
import sys

def analyze_ad_quality(image_path: str):
    """
    Analyze HOW GOOD an ad is (not WHAT it's selling)
    """
    
    print("\n" + "="*70)
    print("🎯 AD QUALITY ANALYSIS SYSTEM")
    print("="*70)
    print("\nEvaluates: How effective is this ad at driving results?")
    print("Focus: Quality metrics, not product guessing")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image not found: {image_path}")
        return None
    
    print(f"\n📸 Analyzing: {os.path.basename(image_path)}")
    print("\n⚙️  Processing...")
    print("   • Extracting visual features")
    print("   • Analyzing messaging elements")
    print("   • Scoring conversion readiness")
    print("   • Calculating quality metrics")
    print("   • Generating recommendations")
    
    try:
        gen = QualityPDFGenerator()
        pdf_path = gen.generate_report(image_path)
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"\n" + "="*70)
            print("✅ SUCCESS!")
            print("="*70)
            print(f"📄 Quality Report: {pdf_path}")
            print(f"📊 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
            print("="*70)
            print("\n📋 Report includes:")
            print("   • Overall quality grade (A-F)")
            print("   • 7 detailed quality metrics")
            print("   • Ad strategy classification")
            print("   • Specific strengths & weaknesses")
            print("   • Prioritized recommendations")
            print("   • Industry benchmark comparison")
            print("\n✨ This report focuses on EFFECTIVENESS, not product type!")
            print("="*70)
            
            return pdf_path
        else:
            print("\n❌ Error: PDF generation failed")
            return None
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_analyze(folder_path: str = "ads", output_folder: str = "quality_reports"):
    """Analyze all ads in a folder"""
    
    import glob
    
    # Find images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(f"{folder_path}/{ext}"))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}/")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📊 BATCH QUALITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Processing {len(image_files)} ads from {folder_path}/")
    print(f"{'='*70}\n")
    
    gen = QualityPDFGenerator()
    
    for i, image_path in enumerate(sorted(image_files), 1):
        ad_name = os.path.basename(image_path)
        pdf_name = os.path.splitext(ad_name)[0] + "_quality.pdf"
        output_pdf = os.path.join(output_folder, pdf_name)
        
        print(f"[{i}/{len(image_files)}] {ad_name}")
        
        try:
            gen.generate_report(image_path, output_pdf)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ Batch complete! Reports saved in: {output_folder}/")
    print(f"📄 Total reports: {len(os.listdir(output_folder))}")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch mode
            folder = sys.argv[2] if len(sys.argv) > 2 else "ads"
            batch_analyze(folder)
        else:
            # Single image
            image_path = sys.argv[1]
            pdf_path = analyze_ad_quality(image_path)
            
            if pdf_path:
                print(f"\n💡 Tip: Open {pdf_path} to see detailed analysis")
    else:
        print("\n" + "="*70)
        print("AD QUALITY ANALYSIS - Usage")
        print("="*70)
        print("\n📍 Analyze single ad:")
        print("   python final_demo.py your_ad.png")
        print("\n📍 Analyze all ads in folder:")
        print("   python final_demo.py --batch ads/")
        print("\n" + "="*70)