"""
Complete Working Backend for Ad Intelligence Platform
Returns proper data structure that frontend expects
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'port': 5001,
        'version': '2.0',
        'features': ['basic_analysis', 'improvement_roadmap', 'ab_tests']
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint
    Returns complete data structure with all features
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get target audience
        target_audience_json = request.form.get('target_audience', '{}')
        target_audience = json.loads(target_audience_json)
        
        # Determine file type
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        is_video = file_ext in ['mp4', 'mov', 'avi', 'webm']
        file_type = 'video' if is_video else 'image'
        
        print(f"\n{'='*60}")
        print(f"📸 Analyzing {file_type}: {file.filename}")
        print(f"👥 Target Audience: {target_audience}")
        print(f"{'='*60}\n")
        
        # TODO: Replace this with actual AI analysis
        # For now, return comprehensive mock data that matches frontend expectations
        
        result = {
            'type': file_type,
            'filename': file.filename,
            'timestamp': datetime.now().isoformat(),
            
            # Summary section
            'summary': {
                'grade': 'B+',
                'overall_score': 7.5,
                'headline': 'Strong visual foundation with optimization opportunities',
                'description': 'This advertisement demonstrates solid creative execution with clear areas for improvement, particularly in call-to-action strength and emotional engagement.',
                'key_insights': [
                    '✅ Strong visual composition with good color contrast',
                    '⚠️ Call-to-action lacks urgency and prominence',
                    '💡 Emotional appeal could be strengthened for better connection',
                    '🎯 Target audience alignment is good but could be refined',
                    '⚡ Adding social proof would boost trust significantly'
                ]
            },
            
            # Detailed features (for bubble map)
            'features': {
                'emotional_signals': {
                    'primary_emotion': 'curiosity',
                    'emotional_intensity': 6,
                    'emotional_authenticity': 7,
                    'aspirational_appeal': 6,
                    'humor_present': False,
                    'creates_fomo': True,
                    'fomo_intensity': 5,
                    'trust_building_elements': 6,
                    'relatability_score': 7,
                    'vulnerability_shown': 4
                },
                'visual_composition': {
                    'visual_complexity': 6,
                    'composition_balance': 8,
                    'rule_of_thirds_adherence': 7,
                    'focal_point_clarity': 8,
                    'color_scheme': 'vibrant',
                    'color_psychology_match': 7,
                    'dominant_colors': ['blue', 'white', 'orange'],
                    'color_harmony': 8,
                    'contrast_level': 8,
                    'whitespace_usage': 7,
                    'visual_hierarchy_clarity': 7,
                    'professional_polish': 8,
                    'production_quality': 8,
                    'lighting_quality': 7
                },
                'engagement_predictors': {
                    'scroll_stopping_power': 7,
                    'first_3_sec_hook': 6,
                    'attention_retention': 6,
                    'curiosity_gap': 7,
                    'social_proof_elements': 4,
                    'scarcity_indicators': 3,
                    'urgency_level': 4,
                    'pattern_interruption': 6,
                    'novelty_factor': 6,
                    'memability': 7,
                    'shareability': 6
                },
                'copy_analysis': {
                    'headline_strength': 6,
                    'headline_clarity': 7,
                    'value_prop_clarity': 6,
                    'benefit_focused': 5,
                    'readability_score': 8,
                    'power_words_used': 5,
                    'call_to_action_present': True,
                    'cta_strength': 5,
                    'cta_specificity': 6,
                    'cta_urgency': 4,
                    'message_clarity': 7
                },
                'predicted_performance': {
                    'estimated_ctr': 7,
                    'estimated_engagement_rate': 7,
                    'estimated_conversion_potential': 6,
                    'virality_potential': 6,
                    'overall_effectiveness': 7
                }
            },
            
            # Critical weaknesses
            'critical_weaknesses': [
                'CTA lacks urgency language (HIGH severity) - No time-sensitive elements to drive action',
                'Limited social proof (MEDIUM severity) - No testimonials, ratings, or user counts visible',
                'Emotional intensity low (MEDIUM severity) - Fails to create strong emotional response'
            ],
            
            # Key strengths  
            'key_strengths': [
                'Visual composition excellent (HIGH impact) - Strong use of color and contrast',
                'Brand clarity strong (HIGH impact) - Logo and branding clearly visible',
                'Professional production (MEDIUM impact) - High-quality imagery and design'
            ],
            
            # Improvement roadmap
            'improvement_roadmap': {
                'quick_wins': [
                    {
                        'action': 'Increase CTA button size by 40% and change to high-contrast color',
                        'impact': 'Expected +12-15% click-through rate increase',
                        'effort': 'low',
                        'priority': 1
                    },
                    {
                        'action': 'Add urgency text: "Limited Time: Save 40%" or "Only 3 Days Left"',
                        'impact': 'Expected +8-10% engagement boost',
                        'effort': 'low',
                        'priority': 1
                    },
                    {
                        'action': 'Move CTA button to upper third of creative for better visibility',
                        'impact': 'Expected +5-7% interaction rate improvement',
                        'effort': 'low',
                        'priority': 2
                    }
                ],
                'medium_term': [
                    {
                        'action': 'Add social proof element: "Join 50,000+ happy customers" with star rating',
                        'impact': 'Expected +15-20% trust and conversion improvement',
                        'effort': 'medium',
                        'priority': 2
                    },
                    {
                        'action': 'Create emotional narrative: Show before/after or customer success story',
                        'impact': 'Expected +20-25% engagement from emotional connection',
                        'effort': 'medium',
                        'priority': 3
                    },
                    {
                        'action': 'A/B test different value propositions in headline',
                        'impact': 'Expected +10-15% CTR from message optimization',
                        'effort': 'medium',
                        'priority': 2
                    }
                ],
                'long_term': [
                    {
                        'action': 'Develop platform-specific variants (Instagram Stories, TikTok, FB Feed)',
                        'impact': 'Expected 30-40% overall performance increase',
                        'effort': 'high',
                        'priority': 4
                    },
                    {
                        'action': 'Create video version with testimonials and product demonstration',
                        'impact': 'Expected 2-3x engagement vs static image',
                        'effort': 'high',
                        'priority': 3
                    }
                ]
            },
            
            # A/B test recommendations
            'ab_test_recommendations': [
                {
                    'test': 'CTA Color Variant',
                    'hypothesis': 'High-contrast orange CTA will outperform current blue by capturing more attention and increasing urgency perception',
                    'expected_lift': '10-15%',
                    'variant_suggestion': 'Change CTA button from #3B82F6 (blue) to #F97316 (orange) with white text'
                },
                {
                    'test': 'Headline: Benefit vs Feature',
                    'hypothesis': 'Benefit-focused headlines ("Save 3 Hours Daily") outperform feature-focused by 15-20% in engagement',
                    'expected_lift': '15-20%',
                    'variant_suggestion': 'Replace current headline with clear benefit statement focusing on time saved or money earned'
                },
                {
                    'test': 'Social Proof Addition',
                    'hypothesis': 'Adding "Join 50,000+ users" badge will increase trust and conversion by leveraging social validation',
                    'expected_lift': '12-18%',
                    'variant_suggestion': 'Add user count badge in top-right corner with 5-star rating display'
                },
                {
                    'test': 'Urgency Language Test',
                    'hypothesis': 'Time-limited offers ("24 Hours Only") create urgency and boost immediate action',
                    'expected_lift': '8-12%',
                    'variant_suggestion': 'Add countdown timer or "Limited Time" banner at top of creative'
                }
            ],
            
            # Executive summary
            'executive_summary': {
                'overall_grade': 'B+',
                'one_sentence_verdict': 'Solid creative foundation with clear optimization path through CTA strengthening, urgency addition, and social proof integration',
                'biggest_opportunity': 'Strengthen call-to-action with high-contrast design, urgency language, and prominent placement to capture 10-15% more conversions',
                'estimated_roi_multiplier': '2-3x'
            }
        }
        
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid JSON in target_audience: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Error in analyze endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """
    Get saved results (for History page)
    Returns empty list for now - localStorage handles this on frontend
    """
    return jsonify([])

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AD INTELLIGENCE BACKEND v2.0")
    print("="*70)
    print(f"   Port: 5001")
    print(f"   Endpoints:")
    print(f"     • GET  /api/health")
    print(f"     • POST /api/analyze")
    print(f"     • GET  /api/results")
    print(f"   Features:")
    print(f"     ✅ Complete data structure")
    print(f"     ✅ Improvement roadmaps")
    print(f"     ✅ A/B test recommendations")
    print(f"     ✅ Executive summaries")
    print("="*70)
    print(f"\n🌐 Backend ready at: http://localhost:5001")
    print(f"📝 Test health: http://localhost:5001/api/health\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)