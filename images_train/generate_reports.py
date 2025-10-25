"""
Generate Human-Readable Reports for All Ads
Creates easy-to-understand analysis reports for each ad image
"""

from human_insights import AdInsightsGenerator
import glob
import json
import os
from pathlib import Path

def generate_all_reports(ads_folder="ads", output_folder="ad_reports"):
    """
    Generate human-readable reports for all ad images
    
    Args:
        ads_folder: Folder containing ad images
        output_folder: Folder to save reports
    """
    
    print("="*80)
    print("GENERATING HUMAN-READABLE AD REPORTS")
    print("="*80)
    
    # Find all image files
    image_patterns = [
        f"{ads_folder}/*.png",
        f"{ads_folder}/*.jpg",
        f"{ads_folder}/*.jpeg",
    ]
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"\n❌ No images found in '{ads_folder}/'")
        return
    
    print(f"\n📁 Found {len(image_files)} images to analyze\n")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize generator
    generator = AdInsightsGenerator()
    
    # Generate reports
    all_insights = []
    
    for i, image_path in enumerate(sorted(image_files), 1):
        ad_name = os.path.basename(image_path)
        print(f"[{i}/{len(image_files)}] Analyzing {ad_name}...")
        
        try:
            # Generate insights
            insights = generator.generate_insights(image_path)
            
            if 'error' not in insights:
                # Save individual report as text file
                report_text = generator.generate_readable_report(image_path)
                report_filename = Path(image_path).stem + "_report.txt"
                report_path = os.path.join(output_folder, report_filename)
                
                with open(report_path, 'w') as f:
                    f.write(report_text)
                
                # Save JSON version too
                json_filename = Path(image_path).stem + "_insights.json"
                json_path = os.path.join(output_folder, json_filename)
                
                with open(json_path, 'w') as f:
                    json.dump(insights, f, indent=2)
                
                all_insights.append(insights)
                print(f"   ✅ Report saved: {report_filename}")
            else:
                print(f"   ❌ Error: {insights['error']}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    # Generate comparative summary
    if all_insights:
        summary = generate_summary_report(all_insights)
        summary_path = os.path.join(output_folder, "SUMMARY_REPORT.txt")
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"\n✅ Summary report saved: SUMMARY_REPORT.txt")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n📂 All reports saved in: {output_folder}/")
    print(f"   • {len(all_insights)} individual ad reports (.txt files)")
    print(f"   • {len(all_insights)} detailed insights (.json files)")
    print(f"   • 1 summary comparison report")
    print(f"\n💡 Open any .txt file to read the human-friendly analysis!")


def generate_summary_report(all_insights: list) -> str:
    """Generate a comparative summary of all ads"""
    
    report = []
    report.append("="*80)
    report.append("CAMPAIGN ANALYSIS SUMMARY - ALL ADS")
    report.append("="*80)
    
    # Overall stats
    report.append(f"\n📊 DATASET OVERVIEW")
    report.append("-"*80)
    report.append(f"Total ads analyzed: {len(all_insights)}")
    
    # Count target audiences
    audiences = {}
    for insight in all_insights:
        demo = insight['target_audience']['primary_demographic'][0] if insight['target_audience']['primary_demographic'] else 'Unknown'
        audiences[demo] = audiences.get(demo, 0) + 1
    
    report.append(f"\nTarget audience distribution:")
    for audience, count in sorted(audiences.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  • {audience}: {count} ads")
    
    # Count product types
    products = {}
    for insight in all_insights:
        cat = insight['product_type']['category']
        products[cat] = products.get(cat, 0) + 1
    
    report.append(f"\nProduct/service types:")
    for product, count in sorted(products.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  • {product}: {count} ads")
    
    # Performance analysis
    report.append(f"\n\n📈 PERFORMANCE ANALYSIS")
    report.append("-"*80)
    
    # Top performers
    sorted_by_performance = sorted(all_insights, 
                                   key=lambda x: x['engagement_prediction']['performance_score'],
                                   reverse=True)
    
    report.append(f"\n🏆 TOP 3 HIGHEST PERFORMING ADS:")
    for i, insight in enumerate(sorted_by_performance[:3], 1):
        ad_name = os.path.basename(insight['image_path'])
        score = insight['engagement_prediction']['performance_score']
        report.append(f"\n  #{i} {ad_name}")
        report.append(f"     Performance score: {score}/1.0")
        report.append(f"     Summary: {insight['summary']}")
        report.append(f"     Target: {insight['target_audience']['description']}")
    
    # Bottom performers
    report.append(f"\n\n⚠️ NEEDS IMPROVEMENT (Bottom 3):")
    for i, insight in enumerate(sorted_by_performance[-3:], 1):
        ad_name = os.path.basename(insight['image_path'])
        score = insight['engagement_prediction']['performance_score']
        report.append(f"\n  {ad_name}")
        report.append(f"     Performance score: {score}/1.0")
        if insight['weaknesses']:
            report.append(f"     Main issue: {insight['weaknesses'][0]}")
    
    # Common strengths
    report.append(f"\n\n✅ COMMON STRENGTHS ACROSS CAMPAIGN:")
    report.append("-"*80)
    all_strengths = {}
    for insight in all_insights:
        for strength in insight['strengths']:
            # Extract key phrase
            key = strength.split('-')[0].strip()
            all_strengths[key] = all_strengths.get(key, 0) + 1
    
    for strength, count in sorted(all_strengths.items(), key=lambda x: x[1], reverse=True)[:5]:
        report.append(f"  • {strength} ({count} ads)")
    
    # Common weaknesses
    report.append(f"\n\n⚠️ COMMON WEAKNESSES TO ADDRESS:")
    report.append("-"*80)
    all_weaknesses = {}
    for insight in all_insights:
        for weakness in insight['weaknesses']:
            # Extract key phrase
            key = weakness.split('-')[0].strip()
            all_weaknesses[key] = all_weaknesses.get(key, 0) + 1
    
    for weakness, count in sorted(all_weaknesses.items(), key=lambda x: x[1], reverse=True)[:5]:
        report.append(f"  • {weakness} ({count} ads)")
    
    # Overall recommendations
    report.append(f"\n\n💡 CAMPAIGN-WIDE RECOMMENDATIONS:")
    report.append("-"*80)
    
    # Calculate campaign stats
    avg_scrollability = sum(i['technical_features'].get('scrollability_score', 0) for i in all_insights) / len(all_insights)
    has_cta_count = sum(1 for i in all_insights if i['technical_features'].get('has_cta', False))
    has_faces_count = sum(1 for i in all_insights if i['technical_features'].get('face_count', 0) > 0)
    
    if avg_scrollability < 0.6:
        report.append("  ➤ Overall scrollability is below optimal - increase contrast and visual interest")
    
    if has_cta_count < len(all_insights) * 0.7:
        report.append(f"  ➤ Only {has_cta_count}/{len(all_insights)} ads have CTAs - add clear calls-to-action")
    
    if has_faces_count < len(all_insights) * 0.4:
        report.append(f"  ➤ Only {has_faces_count}/{len(all_insights)} ads feature people - consider adding human elements")
    
    report.append(f"\n  ➤ Focus optimization efforts on the bottom 3 performing ads")
    report.append(f"  ➤ Replicate successful elements from top performers across campaign")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Change these paths if needed
    ads_folder = "images_train/image_ads"  # Folder with your ad images
    output_folder = "ad_reports"  # Where to save reports
    
    generate_all_reports(ads_folder, output_folder)