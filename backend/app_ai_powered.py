"""
Flask Backend for Ad Intelligence Platform - AI-POWERED VERSION
Uses OpenAI through Lava to generate ALL recommendations dynamically
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

# Import our AI-powered analyzer
from lava_extractor_ai_powered import LavaAdFeatureExtractor

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

# Initialize AI-powered analyzer
image_analyzer = LavaAdFeatureExtractor(model='gpt-4o-mini', max_workers=1)

print("\n" + "="*70)
print("✅ AI-POWERED AD INTELLIGENCE BACKEND INITIALIZED")
print("="*70)
print("Using OpenAI through Lava for:")
print("  • Feature extraction")
print("  • Improvement roadmaps (AI-generated)")
print("  • A/B test recommendations (AI-generated)")
print("  • Strategic insights")
print("="*70 + "\n")


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
        'service': 'AI-Powered Ad Intelligence API',
        'version': '2.0.0',
        'ai_features': {
            'roadmap_generation': 'AI-powered',
            'ab_testing': 'AI-powered',
            'recommendations': 'AI-powered'
        }
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_ad():
    """
    Main endpoint to analyze image or video ad with AI-generated recommendations
    
    Expects: multipart/form-data with 'file' field
    Returns: JSON with AI-generated features, roadmaps, and A/B tests
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
        
        print(f"\n📁 File saved: {filepath}")
        
        # Get target audience if provided
        target_audience_json = request.form.get('target_audience', '{}')
        target_audience = json.loads(target_audience_json)
        
        # Analyze based on file type
        if is_video(filename):
            print("🎬 Video analysis not yet supported with AI recommendations")
            return jsonify({'error': 'Video analysis coming soon'}), 501
        else:
            print("📸 Analyzing image with AI-powered recommendations...")
            result = analyze_image_file(filepath)
        
        # Add metadata
        result['metadata'] = {
            'filename': filename,
            'upload_timestamp': timestamp,
            'file_type': 'video' if is_video(filename) else 'image',
            'target_audience': target_audience,
            'ai_powered': True
        }
        
        # Save result
        result_filename = f"{timestamp}_result.json"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ AI-powered analysis complete!\n")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def analyze_image_file(filepath):
    """Analyze image ad with AI-generated recommendations"""
    
    print("🤖 Using AI to generate:")
    print("   • Core feature analysis")
    print("   • Improvement roadmap (AI-generated)")
    print("   • A/B test recommendations (AI-generated)")
    
    # Extract features with AI-generated recommendations
    features = image_analyzer.extract_features(filepath)
    
    # Clean up numpy types
    features = convert_numpy(features)
    
    # Transform to frontend format
    return transform_to_frontend_format(features, 'image')


def transform_to_frontend_format(raw_features, file_type):
    """
    Transform raw AI features into the format expected by frontend
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
    
    # Get AI-generated improvement roadmap (already in features from AI)
    roadmap = raw_features.get('improvement_roadmap', {
        'quick_wins': [],
        'medium_term': [],
        'long_term': []
    })
    
    # Get AI-generated A/B test recommendations (already in features from AI)
    ab_tests = raw_features.get('ab_test_recommendations', [])
    
    # Get executive summary
    exec_summary = raw_features.get('executive_summary', {})
    
    # Structure response for frontend
    return {
        'success': True,
        'type': file_type,
        'filename': raw_features.get('_meta', {}).get('ad_id', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'features': raw_features,
        'critical_weaknesses': raw_features.get('critical_weaknesses', []),
        'key_strengths': raw_features.get('key_strengths', []),
        'improvement_roadmap': roadmap,  # AI-generated
        'ab_test_recommendations': ab_tests,  # AI-generated
        'executive_summary': exec_summary,
        'ai_generated': True  # Flag to indicate AI-generated content
    }


def generate_summary(features, file_type):
    """Generate human-readable summary from AI features"""
    
    # Use executive summary from AI
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
        'A+': 10, 'A': 9.5, 'A-': 9,
        'B+': 8.5, 'B': 8, 'B-': 7.5,
        'C+': 7, 'C': 6.5, 'C-': 6,
        'D+': 5.5, 'D': 5, 'D-': 4.5,
        'F': 4
    }
    return grade_map.get(grade_str, 7.5)


def score_to_grade(score):
    """Convert numeric score to letter grade"""
    if score >= 9.5: return 'A+'
    elif score >= 9: return 'A'
    elif score >= 8.5: return 'A-'
    elif score >= 8: return 'B+'
    elif score >= 7.5: return 'B'
    elif score >= 7: return 'B-'
    elif score >= 6.5: return 'C+'
    elif score >= 6: return 'C'
    elif score >= 5.5: return 'C-'
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
        'C-': 'Needs Improvement',
        'D': 'Poor Performance',
        'F': 'Needs Major Improvement'
    }
    return descriptions.get(grade, 'Unknown')


def generate_insights_from_features(features):
    """Generate key insights from AI feature analysis"""
    insights = []
    
    # Use AI-generated strengths and weaknesses
    if 'key_strengths' in features and features['key_strengths']:
        for strength in features['key_strengths'][:2]:
            insights.append(f"✅ {strength}")
    
    if 'critical_weaknesses' in features and features['critical_weaknesses']:
        for weakness in features['critical_weaknesses'][:2]:
            insights.append(f"⚠️ {weakness}")
    
    # If we have insights, return them
    if insights:
        return insights[:5]
    
    # Fallback: compute from raw features
    engagement = features.get('engagement_predictors', {})
    copy_analysis = features.get('copy_analysis', {})
    visual = features.get('visual_composition', {})
    
    # Scroll-stopping power
    scroll_stop = engagement.get('scroll_stopping_power', 0)
    if scroll_stop >= 7:
        insights.append(f"✅ Strong scroll-stopping power ({scroll_stop}/10)")
    elif scroll_stop < 5:
        insights.append(f"⚠️ Weak scroll-stopping power ({scroll_stop}/10)")
    
    # CTA analysis
    cta_present = copy_analysis.get('call_to_action_present', False)
    if cta_present:
        cta_strength = copy_analysis.get('cta_strength', 0)
        if cta_strength >= 7:
            insights.append(f"✅ Strong call-to-action ({cta_strength}/10)")
        else:
            insights.append(f"⚠️ CTA needs strengthening ({cta_strength}/10)")
    else:
        insights.append("❌ No clear call-to-action")
    
    # Visual quality
    polish = visual.get('professional_polish', 0)
    if polish >= 8:
        insights.append(f"✅ Professional visual quality ({polish}/10)")
    elif polish < 6:
        insights.append(f"⚠️ Visual quality needs improvement ({polish}/10)")
    
    return insights[:5]


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
                    'grade': data.get('summary', {}).get('grade'),
                    'ai_powered': data.get('metadata', {}).get('ai_powered', False)
                })
    
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(results)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AI-POWERED AD INTELLIGENCE API SERVER")
    print("="*70)
    print("\nStarting server on http://localhost:5001")
    print("\nEndpoints:")
    print("  POST /api/analyze - Upload and analyze ad (AI-powered)")
    print("  GET  /api/results - List previous analyses")
    print("  GET  /api/health  - Health check")
    print("\nAI Features:")
    print("  🤖 AI-generated improvement roadmaps")
    print("  🤖 AI-generated A/B test recommendations")
    print("  🤖 Specific, actionable insights")
    print("  🤖 Industry benchmark-based estimates")
    print("\nNo more hardcoded recommendations!")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)