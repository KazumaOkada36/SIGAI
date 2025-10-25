"""
PDF Report Generator
Creates beautiful, professional PDF reports for ad analysis
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from PIL import Image as PILImage
import os
from datetime import datetime
from human_insights import AdInsightsGenerator

class PDFReportGenerator:
    """Generate professional PDF reports for ad analysis"""
    
    def __init__(self):
        self.generator = AdInsightsGenerator()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Create custom text styles for the PDF"""
        
        # Check if styles already exist (to avoid duplicates)
        if 'CustomTitle' not in self.styles:
            # Title style
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'SectionHeader' not in self.styles:
            # Section header style
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold',
                borderWidth=2,
                borderColor=colors.HexColor('#3498db'),
                borderPadding=5,
                backColor=colors.HexColor('#ecf0f1')
            ))
        
        if 'BodyText' not in self.styles:
            # Body text style
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=10,
                alignment=TA_JUSTIFY
            ))
        
        if 'BulletPoint' not in self.styles:
            # Bullet point style
            self.styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#34495e'),
                leftIndent=20,
                spaceAfter=6
            ))
    
    def generate_pdf_report(self, image_path: str, output_pdf: str = None) -> str:
        """
        Generate a professional PDF report for an ad image
        
        Args:
            image_path: Path to the ad image
            output_pdf: Path for output PDF (auto-generated if None)
        
        Returns:
            Path to generated PDF
        """
        
        # Generate insights
        print(f"🔍 Analyzing {os.path.basename(image_path)}...")
        insights = self.generator.generate_insights(image_path)
        
        if 'error' in insights:
            print(f"❌ Error: {insights['error']}")
            return None
        
        # Determine output filename
        if output_pdf is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_pdf = f"{base_name}_analysis.pdf"
        
        print(f"📄 Creating PDF report...")
        
        # Create PDF
        doc = SimpleDocTemplate(
            output_pdf,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        # Build content
        story = []
        
        # Add title
        title = Paragraph("Ad Performance Analysis Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Add metadata
        metadata_text = f"""
        <b>Ad ID:</b> {os.path.basename(image_path)}<br/>
        <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        <b>Performance Score:</b> {insights['engagement_prediction']['performance_score']:.2f}/1.0
        """
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Add image (if it exists and is not too large)
        if os.path.exists(image_path):
            try:
                img = PILImage.open(image_path)
                img_width, img_height = img.size
                
                # Resize if needed (max width 4 inches)
                max_width = 4 * inch
                if img_width > max_width:
                    aspect = img_height / img_width
                    img_width = max_width
                    img_height = max_width * aspect
                else:
                    img_width = img_width * 0.5
                    img_height = img_height * 0.5
                
                # Add image
                img_obj = Image(image_path, width=min(img_width, 4*inch), height=min(img_height, 4*inch))
                story.append(img_obj)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
        
        # Summary Section
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Paragraph(insights['summary'], self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Target Audience Section
        story.append(Paragraph("👥 Target Audience", self.styles['SectionHeader']))
        story.append(Paragraph(insights['target_audience']['description'], self.styles['BodyText']))
        if insights['target_audience']['characteristics']:
            char_text = "<b>Key Characteristics:</b> " + ", ".join(insights['target_audience']['characteristics'])
            story.append(Paragraph(char_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Product/Service Type Section
        story.append(Paragraph("🛍️ Product/Service Analysis", self.styles['SectionHeader']))
        product_text = f"""
        <b>Category:</b> {insights['product_type']['category']}<br/>
        <b>Description:</b> {insights['product_type']['description']}<br/>
        <b>Confidence Level:</b> {insights['product_type']['confidence'].title()}
        """
        if insights['product_type'].get('specific_examples'):
            examples = ", ".join(insights['product_type']['specific_examples'][:3])
            product_text += f"<br/><b>Specific Examples:</b> {examples}"
        story.append(Paragraph(product_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Emotional Appeal Section
        story.append(Paragraph("💭 Emotional Appeal & Psychology", self.styles['SectionHeader']))
        story.append(Paragraph(insights['emotional_appeal']['description'], self.styles['BodyText']))
        emotion_text = f"""
        <b>Primary Emotions:</b> {', '.join(insights['emotional_appeal']['primary_emotions'])}<br/>
        """
        if insights['emotional_appeal'].get('secondary_emotions'):
            emotion_text += f"<b>Secondary Emotions:</b> {', '.join(insights['emotional_appeal']['secondary_emotions'][:3])}<br/>"
        emotion_text += f"<b>Emotional Intensity:</b> {insights['emotional_appeal']['emotional_intensity'].title()}"
        if insights['emotional_appeal'].get('psychological_triggers'):
            triggers = '; '.join(insights['emotional_appeal']['psychological_triggers'][:2])
            emotion_text += f"<br/><b>Psychological Mechanisms:</b> {triggers}"
        story.append(Paragraph(emotion_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Visual Style Section
        story.append(Paragraph("🎨 Visual Style & Design", self.styles['SectionHeader']))
        style_text = f"""
        <b>Overall Style:</b> {insights['visual_style']['overall_feel']}<br/>
        <b>Color Palette:</b> {insights['visual_style']['color_palette']}<br/>
        <b>Composition:</b> {insights['visual_style']['composition']}
        """
        story.append(Paragraph(style_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Engagement Prediction Section
        story.append(Paragraph("📈 Performance Prediction", self.styles['SectionHeader']))
        
        # Create performance table
        perf_data = [
            ['Metric', 'Prediction'],
            ['Scrollability', insights['engagement_prediction']['scrollability']],
            ['Click Likelihood', insights['engagement_prediction']['click_likelihood']],
            ['Overall Performance', insights['engagement_prediction']['overall_performance']],
            ['Performance Score', f"{insights['engagement_prediction']['performance_score']:.2f}/1.0"]
        ]
        
        perf_table = Table(perf_data, colWidths=[2*inch, 4*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(perf_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Strengths Section
        story.append(Paragraph("✅ Key Strengths", self.styles['SectionHeader']))
        for strength in insights['strengths']:
            # Clean up the strength text
            clean_strength = strength.replace('⭐', '•').replace('✓', '•')
            story.append(Paragraph(clean_strength, self.styles['BulletPoint']))
        story.append(Spacer(1, 0.2*inch))
        
        # Weaknesses Section
        story.append(Paragraph("⚠️ Areas for Improvement", self.styles['SectionHeader']))
        for weakness in insights['weaknesses']:
            # Clean up the weakness text
            clean_weakness = weakness.replace('⚠️', '•').replace('✓', '•')
            story.append(Paragraph(clean_weakness, self.styles['BulletPoint']))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations Section
        story.append(Paragraph("💡 Actionable Recommendations", self.styles['SectionHeader']))
        for i, rec in enumerate(insights['recommendations'], 1):
            # Clean up the recommendation text
            clean_rec = rec.replace('➤', f'{i}.').replace('✓', '•')
            story.append(Paragraph(clean_rec, self.styles['BulletPoint']))
        
        # Add footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = "<i>Report generated by Ad Intelligence Feature Extraction System</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ PDF report created: {output_pdf}")
        return output_pdf


def generate_pdf_for_image(image_path: str, output_pdf: str = None):
    """
    Simple function to generate a PDF report for an image
    
    Usage:
        generate_pdf_for_image("ad.png")
        # Creates: ad_analysis.pdf
    """
    generator = PDFReportGenerator()
    return generator.generate_pdf_report(image_path, output_pdf)


def generate_pdfs_for_all_images(ads_folder: str = "ads", output_folder: str = "pdf_reports"):
    """
    Generate PDF reports for all images in a folder
    
    Usage:
        generate_pdfs_for_all_images("ads")
        # Creates pdf_reports/ folder with all PDFs
    """
    import glob
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(f"{ads_folder}/{ext}"))
    
    if not image_files:
        print(f"❌ No images found in {ads_folder}/")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"📊 Generating PDF reports for {len(image_files)} images...\n")
    print("="*60)
    
    generator = PDFReportGenerator()
    
    for i, image_path in enumerate(sorted(image_files), 1):
        ad_name = os.path.basename(image_path)
        pdf_name = os.path.splitext(ad_name)[0] + "_analysis.pdf"
        output_pdf = os.path.join(output_folder, pdf_name)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {ad_name}")
        
        try:
            generator.generate_pdf_report(image_path, output_pdf)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print(f"✅ All PDF reports saved in: {output_folder}/")
    print(f"📄 Total PDFs created: {len(os.listdir(output_folder))}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single image mode
        image_path = sys.argv[1]
        output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
        generate_pdf_for_image(image_path, output_pdf)
    else:
        # Batch mode - process all images in ads folder
        print("Usage:")
        print("  Single image: python pdf_generator.py image.png [output.pdf]")
        print("  All images:   python pdf_generator.py")
        print("\nProcessing all images in 'ads/' folder...\n")
        generate_pdfs_for_all_images()