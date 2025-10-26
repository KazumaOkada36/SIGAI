"""
Flask Backend for Ad Intelligence Platform - FIXED VERSION
Properly transforms AI analysis into frontend-expected format
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Import our analyzers
from lava_extractor_enhanced import LavaAdFeatureExtractor
from video_analyzer import VideoAdAnalyzer

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi', 'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Initialize analyzers
image_analyzer = LavaAdFeatureExtractor(model='gpt-4o-mini', max_workers=1)
video_analyzer = VideoAdAnalyzer(model='gpt-4o-mini')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    video_exts = {'mp4', 'mov', 'avi', 'webm'}
    return filename.rsplit('.', 1)[1].lower() in video_exts


def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


@app.route('/')
def serve_frontend():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Ad Intelligence API',
        'version': '1.0.0'
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_ad():
    """
    Main endpoint to analyze image or video ad
    
    Expects: multipart/form-data with 'file' field
    Returns: JSON with extracted features in frontend-compatible format
    """
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"📁 File saved: {filepath}")
        
        # Get target audience if provided
        target_audience_json = request.form.get('target_audience', '{}')
        target_audience = json.loads(target_audience_json)
        
        # Analyze based on file type
        if is_video(filename):
            print("🎬 Analyzing video...")
            result = analyze_video_file(filepath)
        else:
            print("📸 Analyzing image...")
            result = analyze_image_file(filepath)
        
        # Add metadata
        result['metadata'] = {
            'filename': filename,
            'upload_timestamp': timestamp,
            'file_type': 'video' if is_video(filename) else 'image',
            'target_audience': target_audience
        }
        
        # Save result
        result_filename = f"{timestamp}_result.json"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Analysis complete!")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def analyze_image_file(filepath):
    """Analyze image ad and return frontend-compatible format"""
    
    features = image_analyzer.extract_features(filepath)
    
    # Clean up numpy types
    features = convert_numpy(features)
    
    # Transform to frontend format
    return transform_to_frontend_format(features, 'image')


def analyze_video_file(filepath):
    """Analyze video ad and return frontend-compatible format"""
    
    analysis = video_analyzer.analyze_video(filepath, num_frames='auto')
    
    # Clean up numpy types
    analysis = convert_numpy(analysis)
    
    # Transform to frontend format
    return transform_to_frontend_format(analysis, 'video')


def transform_to_frontend_format(raw_features, file_type):
    """
    Transform raw AI features into the format expected by frontend
    This matches the structure from backend_working.py
    """
    
    if 'error' in raw_features:
        return {
            'success': False,
            'type': file_type,
            'error': raw_features.get('error'),
            'summary': {
                'grade': 'N/A',
                'overall_score': 0,
                'headline': 'Analysis Error',
                'description': raw_features.get('error', 'Unknown error'),
                'key_insights': []
            }
        }
    
    # Generate summary
    summary = generate_summary(raw_features, file_type)
    
    # Get improvement roadmap if available
    roadmap = raw_features.get('improvement_roadmap', generate_default_roadmap(raw_features))
    
    # Get A/B test recommendations
    ab_tests = raw_features.get('ab_test_recommendations', generate_default_ab_tests(raw_features))
    
    # Get executive summary
    exec_summary = raw_features.get('executive_summary', generate_default_exec_summary(raw_features))
    
    # Structure response for frontend
    return {
        'success': True,
        'type': file_type,
        'filename': raw_features.get('_meta', {}).get('ad_id', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'features': raw_features,
        'critical_weaknesses': raw_features.get('critical_weaknesses', extract_weaknesses(raw_features)),
        'key_strengths': raw_features.get('key_strengths', extract_strengths(raw_features)),
        'improvement_roadmap': roadmap,
        'ab_test_recommendations': ab_tests,
        'executive_summary': exec_summary
    }


def generate_summary(features, file_type):
    """Generate human-readable summary from features"""
    
    # Use executive summary if available
    if 'executive_summary' in features:
        exec_sum = features['executive_summary']
        return {
            'grade': exec_sum.get('overall_grade', 'B+'),
            'overall_score': extract_score_from_grade(exec_sum.get('overall_grade', 'B+')),
            'headline': exec_sum.get('one_sentence_verdict', 'Advertisement Analysis Complete'),
            'description': exec_sum.get('biggest_opportunity', 'Analysis shows opportunities for optimization.'),
            'key_insights': generate_insights_from_features(features)
        }
    
    # Fallback: Calculate from raw features
    if file_type == 'image':
        performance = features.get('predicted_performance', {})
        overall_score = performance.get('overall_effectiveness', 7)
    else:
        vl = features.get('video_level_features', {})
        performance = vl.get('predicted_performance', {})
        overall_score = performance.get('estimated_engagement', 7)
    
    grade = score_to_grade(overall_score)
    
    return {
        'grade': grade,
        'overall_score': overall_score,
        'headline': f'Grade {grade} - {get_grade_description(grade)}',
        'description': f'This ad scored {overall_score}/10 for overall effectiveness.',
        'key_insights': generate_insights_from_features(features)
    }


def extract_score_from_grade(grade_str):
    """Convert grade like 'B+' to numeric score"""
    grade_map = {
        'A+': 10, 'A': 9, 'A-': 8.5,
        'B+': 8, 'B': 7.5, 'B-': 7,
        'C+': 6.5, 'C': 6, 'C-': 5.5,
        'D': 5, 'F': 4
    }
    return grade_map.get(grade_str, 7.5)


def score_to_grade(score):
    """Convert numeric score to letter grade"""
    if score >= 9.5: return 'A+'
    elif score >= 9: return 'A'
    elif score >= 8.5: return 'A-'
    elif score >= 8: return 'B+'
    elif score >= 7: return 'B'
    elif score >= 6: return 'C'
    elif score >= 5: return 'D'
    else: return 'F'


def get_grade_description(grade):
    """Get description for grade"""
    descriptions = {
        'A+': 'Outstanding Performance',
        'A': 'Excellent Performance',
        'A-': 'Very Good Performance',
        'B+': 'Good Performance',
        'B': 'Above Average',
        'B-': 'Satisfactory',
        'C+': 'Average Performance',
        'C': 'Below Average',
        'D': 'Poor Performance',
        'F': 'Needs Major Improvement'
    }
    return descriptions.get(grade, 'Unknown')


def generate_insights_from_features(features):
    """Generate key insights from feature analysis"""
    insights = []
    
    # Check for pre-existing insights
    if 'key_strengths' in features and features['key_strengths']:
        for strength in features['key_strengths'][:2]:
            insights.append(f"✅ {strength}")
    
    if 'critical_weaknesses' in features and features['critical_weaknesses']:
        for weakness in features['critical_weaknesses'][:2]:
            insights.append(f"⚠️ {weakness}")
    
    # If we have insights, return them
    if insights:
        return insights[:5]
    
    # Otherwise, compute from raw features
    engagement = features.get('engagement_predictors', {})
    emotional = features.get('emotional_signals', {})
    copy_analysis = features.get('copy_analysis', {})
    visual = features.get('visual_composition', {})
    
    # Scroll-stopping power
    scroll_stop = engagement.get('scroll_stopping_power', 0)
    if scroll_stop >= 7:
        insights.append(f"✅ Strong scroll-stopping power ({scroll_stop}/10)")
    elif scroll_stop < 5:
        insights.append(f"⚠️ Weak scroll-stopping power ({scroll_stop}/10) - needs attention")
    
    # CTA analysis
    cta_present = copy_analysis.get('call_to_action_present', False)
    if cta_present:
        cta_strength = copy_analysis.get('cta_strength', 0)
        if cta_strength >= 7:
            insights.append(f"✅ Clear and strong call-to-action ({cta_strength}/10)")
        else:
            insights.append(f"⚠️ CTA present but lacks urgency ({cta_strength}/10)")
    else:
        insights.append("❌ No clear call-to-action - critical weakness")
    
    # Emotional appeal
    emotion = emotional.get('primary_emotion', 'unknown')
    intensity = emotional.get('emotional_intensity', 0)
    if intensity >= 7:
        insights.append(f"💡 Strong emotional appeal: {emotion} (intensity: {intensity}/10)")
    elif intensity < 5:
        insights.append(f"💡 Weak emotional connection: {emotion} (intensity: {intensity}/10)")
    
    # Visual quality
    polish = visual.get('professional_polish', 0)
    if polish >= 8:
        insights.append(f"✅ Professional visual quality ({polish}/10)")
    elif polish < 6:
        insights.append(f"⚠️ Visual quality needs improvement ({polish}/10)")
    
    return insights[:5]


def extract_weaknesses(features):
    """Extract critical weaknesses if not provided"""
    if 'critical_weaknesses' in features:
        return features['critical_weaknesses']
    
    weaknesses = []
    
    # Check engagement
    engagement = features.get('engagement_predictors', {})
    if engagement.get('scroll_stopping_power', 10) < 5:
        weaknesses.append("Low scroll-stopping power (HIGH severity) - Ad fails to capture attention")
    
    if engagement.get('urgency_level', 10) < 5:
        weaknesses.append("Lacks urgency (MEDIUM severity) - No time-sensitive elements")
    
    # Check copy
    copy_analysis = features.get('copy_analysis', {})
    if not copy_analysis.get('call_to_action_present', True):
        weaknesses.append("No clear CTA (HIGH severity) - Users won't know what action to take")
    elif copy_analysis.get('cta_strength', 10) < 5:
        weaknesses.append("Weak CTA (MEDIUM severity) - Call-to-action lacks clarity or urgency")
    
    # Check social proof
    if engagement.get('social_proof_elements', 10) < 4:
        weaknesses.append("Limited social proof (MEDIUM severity) - Missing testimonials or trust signals")
    
    return weaknesses[:5]


def extract_strengths(features):
    """Extract key strengths if not provided"""
    if 'key_strengths' in features:
        return features['key_strengths']
    
    strengths = []
    
    visual = features.get('visual_composition', {})
    engagement = features.get('engagement_predictors', {})
    copy_analysis = features.get('copy_analysis', {})
    
    if visual.get('professional_polish', 0) >= 7:
        polish = visual.get('professional_polish')
        strengths.append(f"High production quality (HIGH impact) - Professional polish: {polish}/10")
    
    if visual.get('composition_balance', 0) >= 7:
        strengths.append("Strong visual composition (HIGH impact) - Well-balanced design")
    
    if engagement.get('scroll_stopping_power', 0) >= 7:
        power = engagement.get('scroll_stopping_power')
        strengths.append(f"Excellent attention-grabbing (HIGH impact) - Scroll-stop power: {power}/10")
    
    if copy_analysis.get('message_clarity', 0) >= 7:
        strengths.append("Clear messaging (MEDIUM impact) - Value proposition is easy to understand")
    
    return strengths[:5]


def generate_default_roadmap(features):
    """Generate improvement roadmap if not provided"""
    if 'improvement_roadmap' in features:
        return features['improvement_roadmap']
    
    # Generate based on weaknesses
    quick_wins = []
    medium_term = []
    long_term = []
    
    copy_analysis = features.get('copy_analysis', {})
    engagement = features.get('engagement_predictors', {})
    
    # Quick wins
    if copy_analysis.get('cta_strength', 10) < 7:
        quick_wins.append({
            'action': 'Increase CTA button size by 40% and use high-contrast color',
            'impact': 'Expected +12-15% click-through rate increase',
            'effort': 'low',
            'priority': 1
        })
    
    if engagement.get('urgency_level', 10) < 5:
        quick_wins.append({
            'action': 'Add urgency text: "Limited Time" or countdown timer',
            'impact': 'Expected +8-10% engagement boost',
            'effort': 'low',
            'priority': 1
        })
    
    # Medium term
    if engagement.get('social_proof_elements', 10) < 5:
        medium_term.append({
            'action': 'Add social proof: customer count, ratings, or testimonials',
            'impact': 'Expected +15-20% trust and conversion improvement',
            'effort': 'medium',
            'priority': 2
        })
    
    # Long term
    long_term.append({
        'action': 'Develop platform-specific variants (Stories, TikTok, Feed)',
        'impact': 'Expected 30-40% overall performance increase',
        'effort': 'high',
        'priority': 3
    })
    
    return {
        'quick_wins': quick_wins,
        'medium_term': medium_term,
        'long_term': long_term
    }


def generate_default_ab_tests(features):
    """Generate A/B test recommendations if not provided"""
    if 'ab_test_recommendations' in features:
        return features['ab_test_recommendations']
    
    tests = []
    
    # CTA test
    tests.append({
        'test': 'CTA Color Variant',
        'hypothesis': 'High-contrast orange CTA will outperform current by capturing more attention',
        'expected_lift': '10-15%',
        'variant_suggestion': 'Change CTA button to #F97316 (orange) with white text'
    })
    
    # Headline test
    tests.append({
        'test': 'Headline: Benefit vs Feature',
        'hypothesis': 'Benefit-focused headlines outperform feature-focused by creating emotional connection',
        'expected_lift': '15-20%',
        'variant_suggestion': 'Replace with clear benefit statement focusing on outcome'
    })
    
    return tests


def generate_default_exec_summary(features):
    """Generate executive summary if not provided"""
    if 'executive_summary' in features:
        return features['executive_summary']
    
    performance = features.get('predicted_performance', {})
    score = performance.get('overall_effectiveness', 7)
    grade = score_to_grade(score)
    
    return {
        'overall_grade': grade,
        'one_sentence_verdict': f'Solid creative foundation with optimization opportunities',
        'biggest_opportunity': 'Strengthen call-to-action with high-contrast design and urgency language',
        'estimated_roi_multiplier': '2-3x'
    }


@app.route('/api/results/<result_id>', methods=['GET'])
def get_result(result_id):
    """Get a previous analysis result"""
    result_path = os.path.join(RESULTS_FOLDER, f"{result_id}_result.json")
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'Result not found'}), 404
    
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return jsonify(result)


@app.route('/api/results', methods=['GET'])
def list_results():
    """List all previous analysis results"""
    results = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith('_result.json'):
            filepath = os.path.join(RESULTS_FOLDER, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append({
                    'id': filename.replace('_result.json', ''),
                    'timestamp': data.get('metadata', {}).get('upload_timestamp'),
                    'filename': data.get('metadata', {}).get('filename'),
                    'type': data.get('metadata', {}).get('file_type'),
                    'grade': data.get('summary', {}).get('grade')
                })
    
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(results)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AD INTELLIGENCE API SERVER - FIXED VERSION")
    print("="*70)
    print("\nStarting server on http://localhost:5001")
    print("\nEndpoints:")
    print("  POST /api/analyze - Upload and analyze ad")
    print("  GET  /api/results - List previous analyses")
    print("  GET  /api/health  - Health check")
    print("\nFeatures:")
    print("  ✅ Proper data transformation")
    print("  ✅ Frontend-compatible format")
    print("  ✅ Automatic fallbacks")
    print("  ✅ Comprehensive insights")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)