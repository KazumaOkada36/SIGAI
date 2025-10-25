"""
SIMPLE ALL-IN-ONE SCRIPT
User inputs image → Gets PDF report

Perfect for your hackathon demo!
"""

from pdf_generator import generate_pdf_for_image
import os
import sys

def analyze_ad_and_generate_pdf(image_path: str):
    """
    Simple function: Input image → Output PDF
    
    This is what you'd call from your user interface
    
    Args:
        image_path: Path to uploaded ad image
    
    Returns:
        Path to generated PDF report
    """
    
    print("\n" + "="*60)
    print("🎯 AD ANALYSIS SYSTEM")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    # Check if it's an image
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif')
    if not image_path.lower().endswith(valid_extensions):
        print(f"❌ Error: Not a valid image file")
        print(f"   Supported formats: {', '.join(valid_extensions)}")
        return None
    
    print(f"\n📸 Input: {os.path.basename(image_path)}")
    print("\n🔍 Step 1/3: Extracting visual features...")
    print("   • Color analysis")
    print("   • Text detection (OCR)")
    print("   • Face detection")
    print("   • Composition analysis")
    
    print("\n🧠 Step 2/3: Generating insights...")
    print("   • Identifying target audience")
    print("   • Analyzing emotional appeal")
    print("   • Predicting engagement")
    print("   • Creating recommendations")
    
    print("\n📄 Step 3/3: Creating PDF report...")
    
    try:
        # Generate the PDF
        pdf_path = generate_pdf_for_image(image_path)
        
        if pdf_path and os.path.exists(pdf_path):
            print(f"\n✅ SUCCESS!")
            print("="*60)
            print(f"📄 PDF Report: {pdf_path}")
            print(f"📊 File size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
            print("="*60)
            print("\n💡 You can now:")
            print(f"   • Open the PDF: {pdf_path}")
            print(f"   • Share with your team")
            print(f"   • Present to judges")
            print(f"   • Email to clients")
            
            return pdf_path
        else:
            print("\n❌ Error: PDF generation failed")
            return None
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        return None


def interactive_mode():
    """Interactive mode - asks user for image path"""
    
    print("\n" + "="*60)
    print("🎨 AD INTELLIGENCE SYSTEM - Interactive Mode")
    print("="*60)
    print("\nThis tool analyzes ad images and generates PDF reports")
    print("with insights about target audience, emotional appeal,")
    print("engagement predictions, and recommendations.")
    print("\n" + "="*60)
    
    while True:
        print("\n")
        image_path = input("📁 Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not image_path:
            print("❌ Please enter a valid path")
            continue
        
        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"').strip("'")
        
        # Analyze and generate PDF
        pdf_path = analyze_ad_and_generate_pdf(image_path)
        
        if pdf_path:
            print("\n" + "="*60)
            choice = input("\nAnalyze another image? (yes/no): ").strip().lower()
            if choice not in ['yes', 'y']:
                print("\n👋 Thanks for using Ad Intelligence System!")
                break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode: python simple_demo.py image.png
        image_path = sys.argv[1]
        pdf_path = analyze_ad_and_generate_pdf(image_path)
        
        if pdf_path:
            print(f"\n✨ Open your report: {pdf_path}")
    
    else:
        # Interactive mode
        interactive_mode()