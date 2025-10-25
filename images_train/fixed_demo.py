"""
FIXED DEMO - Uses improved insights that actually work
"""

from improved_insights import ImprovedAdInsightsGenerator
from pdf_generator import PDFReportGenerator
import os
import sys

def analyze_ad_fixed(image_path: str):
    """
    Fixed analysis that actually gets it right
    """
    
    print("\n" + "="*60)
    print("🎯 AD ANALYSIS SYSTEM (FIXED VERSION)")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    print(f"\n📸 Input: {os.path.basename(image_path)}")
    print("\n🔍 Analyzing with improved detection...")
    
    try:
        # Generate insights with improved logic
        gen = ImprovedAdInsightsGenerator()
        insights = gen.generate_insights(image_path)
        
        if 'error' in insights:
            print(f"❌ Error: {insights['error']}")
            return None
        
        # Create PDF using the pdf_generator but with improved insights
        print("\n📄 Creating PDF report...")
        
        # Generate PDF filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = f"{base_name}_analysis_fixed.pdf"
        
        # Use PDFReportGenerator but inject our improved insights
        pdf_gen = PDFReportGenerator()
        
        # Temporarily replace the generator
        old_generator = pdf_gen.generator
        pdf_gen.generator = gen
        
        pdf_gen.generate_pdf_report(image_path, pdf_path)
        
        # Restore
        pdf_gen.generator = old_generator
        
        print(f"\n✅ SUCCESS!")
        print("="*60)
        print(f"📄 PDF Report: {pdf_path}")
        print("="*60)
        
        # Show quick summary
        print("\n📊 Quick Summary:")
        print(f"   Product: {insights['product_type']['category']}")
        print(f"   Audience: {insights['target_audience']['primary_demographic'][0] if insights['target_audience']['primary_demographic'] else 'Unknown'}")
        print(f"   Score: {insights['engagement_prediction']['performance_score']}/1.0")
        
        return pdf_path
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        pdf_path = analyze_ad_fixed(image_path)
        
        if pdf_path:
            print(f"\n✨ Open your report: {pdf_path}")
    else:
        print("Usage: python fixed_demo.py image.png")