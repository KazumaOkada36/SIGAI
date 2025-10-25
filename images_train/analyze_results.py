"""
Analyze Your Extracted Features
View insights and create visualizations from your feature extraction
"""

import pandas as pd
import os

def analyze_features(csv_file="real_ad_features.csv"):
    """Analyze the extracted features and show insights"""
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("   Run 'process_real_ads.py' first to generate features")
        return
    
    df = pd.read_csv(csv_file)
    
    print("="*60)
    print("FEATURE ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total ads analyzed: {len(df)}")
    print(f"   • Total features: {len(df.columns)}")
    
    # 1. ENGAGEMENT ANALYSIS
    print("\n" + "="*60)
    print("1. ENGAGEMENT PREDICTIONS")
    print("="*60)
    
    if 'scrollability_score' in df.columns:
        avg_scroll = df['scrollability_score'].mean()
        high_scroll = len(df[df['scrollability_score'] > 0.7])
        
        print(f"\n📱 Scrollability (will users stop scrolling?):")
        print(f"   • Average: {avg_scroll:.3f}")
        print(f"   • High performers (>0.7): {high_scroll}/{len(df)} ads")
        print(f"   • Best ad: {df.loc[df['scrollability_score'].idxmax(), 'image_path']}")
        print(f"     Score: {df['scrollability_score'].max():.3f}")
    
    if 'curiosity_gap_score' in df.columns:
        avg_curiosity = df['curiosity_gap_score'].mean()
        print(f"\n🤔 Curiosity Gap (creates unanswered questions?):")
        print(f"   • Average: {avg_curiosity:.3f}")
        print(f"   • High curiosity ads (>0.6): {len(df[df['curiosity_gap_score'] > 0.6])}/{len(df)}")
    
    if 'value_prop_clarity' in df.columns:
        avg_clarity = df['value_prop_clarity'].mean()
        print(f"\n💡 Value Prop Clarity (understandable in 2 seconds?):")
        print(f"   • Average: {avg_clarity:.3f}")
        print(f"   • Clear ads (>0.8): {len(df[df['value_prop_clarity'] > 0.8])}/{len(df)}")
    
    # 2. VISUAL ANALYSIS
    print("\n" + "="*60)
    print("2. VISUAL CHARACTERISTICS")
    print("="*60)
    
    if 'face_count' in df.columns:
        total_faces = df['face_count'].sum()
        ads_with_faces = len(df[df['face_count'] > 0])
        print(f"\n👥 Human Presence:")
        print(f"   • Ads with faces: {ads_with_faces}/{len(df)}")
        print(f"   • Total faces detected: {int(total_faces)}")
        print(f"   • Average faces per ad: {df['face_count'].mean():.1f}")
    
    if 'avg_brightness' in df.columns:
        print(f"\n💡 Brightness:")
        print(f"   • Average: {df['avg_brightness'].mean():.1f}")
        print(f"   • Dark ads (<100): {len(df[df['avg_brightness'] < 100])}")
        print(f"   • Bright ads (>150): {len(df[df['avg_brightness'] > 150])}")
    
    if 'visual_complexity' in df.columns:
        print(f"\n🎨 Visual Complexity:")
        print(f"   • Average: {df['visual_complexity'].mean():.1f}")
        print(f"   • Simple (<30): {len(df[df['visual_complexity'] < 30])}")
        print(f"   • Complex (>60): {len(df[df['visual_complexity'] > 60])}")
    
    # 3. TEXT & CTA ANALYSIS
    print("\n" + "="*60)
    print("3. TEXT & CALL-TO-ACTION")
    print("="*60)
    
    if 'has_cta' in df.columns:
        cta_count = df['has_cta'].sum()
        print(f"\n📢 Call-to-Action:")
        print(f"   • Ads with CTA: {cta_count}/{len(df)}")
        print(f"   • Percentage: {cta_count/len(df)*100:.1f}%")
    
    if 'text_element_count' in df.columns:
        print(f"\n📝 Text Elements:")
        print(f"   • Average per ad: {df['text_element_count'].mean():.1f}")
        print(f"   • No text: {len(df[df['text_element_count'] == 0])}")
        print(f"   • Heavy text (>7): {len(df[df['text_element_count'] > 7])}")
    
    if 'has_price' in df.columns:
        print(f"\n💰 Pricing:")
        print(f"   • Ads showing price: {df['has_price'].sum()}/{len(df)}")
    
    if 'urgency_language_score' in df.columns:
        urgent_ads = len(df[df['urgency_language_score'] > 0])
        print(f"\n⚡ Urgency Language:")
        print(f"   • Ads with urgency: {urgent_ads}/{len(df)}")
    
    # 4. MOBILE OPTIMIZATION
    print("\n" + "="*60)
    print("4. MOBILE OPTIMIZATION")
    print("="*60)
    
    if 'mobile_readability_score' in df.columns:
        avg_mobile = df['mobile_readability_score'].mean()
        mobile_friendly = len(df[df['mobile_readability_score'] > 0.7])
        print(f"\n📱 Mobile Readability:")
        print(f"   • Average score: {avg_mobile:.2f}")
        print(f"   • Mobile-friendly (>0.7): {mobile_friendly}/{len(df)}")
    
    if 'hd_quality' in df.columns:
        hd_ads = df['hd_quality'].sum()
        print(f"\n📺 HD Quality:")
        print(f"   • HD resolution ads: {int(hd_ads)}/{len(df)}")
    
    # 5. CREATIVE INSIGHTS
    print("\n" + "="*60)
    print("5. CREATIVE INSIGHTS")
    print("="*60)
    
    if 'information_density' in df.columns:
        density_counts = df['information_density'].value_counts()
        print(f"\n📊 Information Density:")
        for density, count in density_counts.items():
            print(f"   • {density.capitalize()}: {count}/{len(df)}")
    
    if 'brand_product_balance' in df.columns:
        balance_counts = df['brand_product_balance'].value_counts()
        print(f"\n⚖️ Brand vs Product Focus:")
        for balance, count in balance_counts.items():
            print(f"   • {balance.replace('_', ' ').title()}: {count}/{len(df)}")
    
    # 6. TOP PERFORMERS
    print("\n" + "="*60)
    print("6. TOP PERFORMING ADS")
    print("="*60)
    
    # Create composite engagement score
    if all(col in df.columns for col in ['scrollability_score', 'value_prop_clarity', 'first_impression_score']):
        df['engagement_score'] = (
            df['scrollability_score'] * 0.4 +
            df['value_prop_clarity'] * 0.3 +
            df['first_impression_score'] * 0.3
        )
        
        print("\n🏆 Top 3 Most Engaging Ads (composite score):")
        top3 = df.nlargest(3, 'engagement_score')
        
        for i, (idx, row) in enumerate(top3.iterrows(), 1):
            ad_name = os.path.basename(row['image_path'])
            print(f"\n   #{i} {ad_name}")
            print(f"      Engagement Score: {row['engagement_score']:.3f}")
            print(f"      • Scrollability: {row['scrollability_score']:.3f}")
            print(f"      • Value Clarity: {row['value_prop_clarity']:.3f}")
            print(f"      • First Impression: {row['first_impression_score']:.3f}")
            print(f"      • Faces: {row.get('face_count', 0)}")
            print(f"      • Has CTA: {row.get('has_cta', False)}")
    
    # 7. RECOMMENDATIONS
    print("\n" + "="*60)
    print("7. RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    
    recommendations = []
    
    if 'has_cta' in df.columns and df['has_cta'].sum() < len(df) * 0.5:
        recommendations.append("⚠️  Only {:.0f}% of ads have CTAs - adding clear CTAs can increase CTR 2-3x".format(
            df['has_cta'].sum()/len(df)*100))
    
    if 'face_count' in df.columns and df['face_count'].sum() == 0:
        recommendations.append("⚠️  No ads feature human faces - adding faces can boost engagement 20-40%")
    
    if 'mobile_readability_score' in df.columns:
        poor_mobile = len(df[df['mobile_readability_score'] < 0.5])
        if poor_mobile > len(df) * 0.3:
            recommendations.append(f"⚠️  {poor_mobile} ads have poor mobile readability - optimize for mobile (80% of views)")
    
    if 'scrollability_score' in df.columns:
        low_scroll = len(df[df['scrollability_score'] < 0.5])
        if low_scroll > 0:
            recommendations.append(f"⚠️  {low_scroll} ads have low scrollability - increase contrast, faces, or central focus")
    
    if recommendations:
        print("\n")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("\n   ✅ All ads are performing well across key metrics!")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nFull dataset: {csv_file}")
    print("Use this analysis for your hackathon presentation!")


if __name__ == "__main__":
    analyze_features("real_ad_features.csv")