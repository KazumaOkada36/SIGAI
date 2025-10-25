"""
Quality-Focused PDF Report Generator
Shows HOW GOOD the ad is, not what it's selling
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image as PILImage
import os
from datetime import datetime
from ad_quality_scorer import AdQualityScorer

class QualityPDFGenerator:
    """Generate quality-focused PDF reports"""
    
    def __init__(self):
        self.scorer = AdQualityScorer()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Create custom styles"""
        
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=26,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'GradeStyle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='GradeStyle',
                parent=self.styles['Heading1'],
                fontSize=48,
                textColor=colors.HexColor('#2ecc71'),
                spaceAfter=10,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
                spaceBefore=15,
                fontName='Helvetica-Bold',
                borderWidth=2,
                borderColor=colors.HexColor('#3498db'),
                borderPadding=5,
                backColor=colors.HexColor('#ecf0f1')
            ))
        
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=10,
                alignment=TA_JUSTIFY
            ))
        
        if 'BulletPoint' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#34495e'),
                leftIndent=20,
                spaceAfter=6
            ))
    
    def generate_report(self, image_path: str, output_pdf: str = None) -> str:
        """Generate quality-focused PDF report"""
        
        print(f"🔍 Scoring ad quality for {os.path.basename(image_path)}...")
        report = self.scorer.score_ad(image_path)
        
        if 'error' in report:
            print(f"❌ Error: {report['error']}")
            return None
        
        if output_pdf is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_pdf = f"{base_name}_quality_report.pdf"
        
        print(f"📄 Creating PDF report...")
        
        doc = SimpleDocTemplate(
            output_pdf,
            pagesize=letter,
            rightMargin=inch*0.75,
            leftMargin=inch*0.75,
            topMargin=inch*0.75,
            bottomMargin=inch*0.75
        )
        
        story = []
        
        # Title
        title = Paragraph("Ad Quality & Effectiveness Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Overall Grade (BIG and prominent)
        grade = report['overall_grade']['letter_grade']
        grade_color = self._get_grade_color(grade)
        grade_style = ParagraphStyle(
            'TempGrade',
            parent=self.styles['GradeStyle'],
            textColor=grade_color
        )
        grade_text = Paragraph(f"<b>GRADE: {grade}</b>", grade_style)
        story.append(grade_text)
        
        score_text = Paragraph(
            f"<b>Overall Score: {report['overall_grade']['numeric_score']}/100</b>",
            self.styles['BodyText']
        )
        story.append(score_text)
        story.append(Paragraph(report['overall_grade']['description'], self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata_text = f"""
        <b>Ad ID:</b> {os.path.basename(image_path)}<br/>
        <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        story.append(Paragraph(metadata_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add image
        if os.path.exists(image_path):
            try:
                img = PILImage.open(image_path)
                img_width, img_height = img.size
                aspect = img_height / img_width
                
                max_width = 4 * inch
                display_width = min(max_width, img_width * 0.4)
                display_height = display_width * aspect
                
                img_obj = Image(image_path, width=display_width, height=min(display_height, 3*inch))
                story.append(img_obj)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
        
        # Ad Type Classification
        story.append(Paragraph("📊 Ad Strategy Classification", self.styles['SectionHeader']))
        ad_type = report['ad_type']
        ad_type_text = f"""
        <b>Primary Type:</b> {ad_type['primary_type']}<br/>
        <b>Confidence:</b> {ad_type['confidence'].title()}<br/>
        <b>Description:</b> {ad_type['description']}
        """
        if ad_type['tactics_detected']:
            tactics = ", ".join(ad_type['tactics_detected'])
            ad_type_text += f"<br/><b>Tactics Detected:</b> {tactics}"
        story.append(Paragraph(ad_type_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Quality Scores Breakdown
        story.append(Paragraph("📈 Quality Metrics Breakdown", self.styles['SectionHeader']))
        
        scores = report['quality_scores']
        score_data = [
            ['Metric', 'Score', 'Assessment'],
            ['Attention Grab', f"{scores['attention_score']}/100", self._assess_score(scores['attention_score'])],
            ['Message Clarity', f"{scores['clarity_score']}/100", self._assess_score(scores['clarity_score'])],
            ['Urgency/FOMO', f"{scores['urgency_score']}/100", self._assess_score(scores['urgency_score'])],
            ['Conversion Ready', f"{scores['conversion_readiness']}/100", self._assess_score(scores['conversion_readiness'])],
            ['Emotional Impact', f"{scores['emotional_impact']}/100", self._assess_score(scores['emotional_impact'])],
            ['Mobile Optimized', f"{scores['mobile_optimization']}/100", self._assess_score(scores['mobile_optimization'])],
            ['Professional Quality', f"{scores['professional_quality']}/100", self._assess_score(scores['professional_quality'])],
        ]
        
        score_table = Table(score_data, colWidths=[2.2*inch, 1.2*inch, 1.8*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(score_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Visual Analysis
        story.append(Paragraph("🎨 Visual Design Analysis", self.styles['SectionHeader']))
        visual = report['visual_analysis']
        visual_text = f"""
        <b>Brightness:</b> {visual['brightness_level'].title()}<br/>
        <b>Color Saturation:</b> {visual['color_saturation'].title()}<br/>
        <b>Visual Complexity:</b> {visual['visual_complexity'].title()}<br/>
        <b>Human Element:</b> {visual['face_count']} face(s) present<br/>
        <b>Assessment:</b> {visual['assessment']}
        """
        story.append(Paragraph(visual_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Messaging Analysis
        story.append(Paragraph("💬 Messaging & Copy Analysis", self.styles['SectionHeader']))
        messaging = report['messaging_analysis']
        messaging_text = f"""
        <b>Call-to-Action:</b> {'Present ✓' if messaging['has_call_to_action'] else 'Missing ✗'}<br/>
        <b>Urgency Language:</b> {messaging['urgency_intensity'].title()}<br/>
        <b>Pricing Shown:</b> {'Yes' if messaging['has_pricing'] else 'No'}<br/>
        <b>Discount Offer:</b> {'Yes' if messaging['has_discount'] else 'No'}<br/>
        <b>Value Proposition:</b> {messaging['value_proposition_clarity'].title()}<br/>
        <b>Assessment:</b> {messaging['assessment']}
        """
        story.append(Paragraph(messaging_text, self.styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # Strengths
        story.append(Paragraph("✅ Key Strengths", self.styles['SectionHeader']))
        for strength in report['strengths']:
            clean_strength = strength.replace('⭐', '•')
            story.append(Paragraph(clean_strength, self.styles['BulletPoint']))
        story.append(Spacer(1, 0.2*inch))
        
        # Weaknesses
        story.append(Paragraph("⚠️ Areas Needing Improvement", self.styles['SectionHeader']))
        for weakness in report['weaknesses']:
            clean_weakness = weakness.replace('⚠️', '•')
            story.append(Paragraph(clean_weakness, self.styles['BulletPoint']))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations (prioritized)
        story.append(Paragraph("💡 Prioritized Recommendations", self.styles['SectionHeader']))
        for i, rec in enumerate(report['recommendations'], 1):
            clean_rec = rec.replace('🔴', '').replace('🟠', '').replace('🟡', '').replace('🔵', '')
            story.append(Paragraph(f"<b>{i}.</b> {clean_rec}", self.styles['BulletPoint']))
        
        # Benchmark Comparison
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("📊 Industry Benchmark Comparison", self.styles['SectionHeader']))
        
        benchmark = report['benchmark_comparison']
        benchmark_data = [['Element', 'Your Ad', 'Industry Standard', 'Status']]
        
        for key, value in benchmark.items():
            benchmark_data.append([
                key.replace('_', ' ').title(),
                str(value['your_ad']),
                value['industry_standard'],
                value['assessment']
            ])
        
        benchmark_table = Table(benchmark_data, colWidths=[1.5*inch, 1.3*inch, 1.7*inch, 1.3*inch])
        benchmark_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(benchmark_table)
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        footer_text = "<i>Report generated by Ad Intelligence Quality Scoring System</i>"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ Quality report created: {output_pdf}")
        return output_pdf
    
    def _get_grade_color(self, grade: str):
        """Get color for grade"""
        colors_map = {
            'A': colors.HexColor('#27ae60'),  # Green
            'B': colors.HexColor('#2ecc71'),  # Light green
            'C': colors.HexColor('#f39c12'),  # Orange
            'D': colors.HexColor('#e67e22'),  # Dark orange
            'F': colors.HexColor('#e74c3c')   # Red
        }
        return colors_map.get(grade, colors.grey)
    
    def _assess_score(self, score: int) -> str:
        """Assess a score"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        else:
            return 'Needs Work'


if __name__ == "__main__":
    import sys
    
    gen = QualityPDFGenerator()
    
    if len(sys.argv) > 1:
        pdf = gen.generate_report(sys.argv[1])
        if pdf:
            print(f"\n✨ Open: {pdf}")
    else:
        print("Usage: python quality_pdf_generator.py image.png")