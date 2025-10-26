"""
Flask Backend for Ad Intelligence Platform
Handles image and video upload + analysis
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
import base64
from datetime import datetime
import cv2
import numpy as np

# Import our analyzers
from lava_extractor_enhanced import LavaAdFeatureExtractor
from video_analyzer import VideoAdAnalyzer

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for development

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
    Returns: JSON with extracted features
    """
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: images (png, jpg, jpeg, gif) and videos (mp4, mov, avi, webm)'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"📁 File saved: {filepath}")
        
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
            'file_type': 'video' if is_video(filename) else 'image'
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
    """Analyze image ad"""
    
    features = image_analyzer.extract_features(filepath)
    
    # Clean up numpy types
    features = convert_numpy(features)
    
    # Structure response for frontend
    return {
        'success': True,
        'type': 'image',
        'features': features,
        'summary': generate_summary(features)
    }


def analyze_video_file(filepath):
    """Analyze video ad"""
    
    analysis = video_analyzer.analyze_video(filepath, num_frames='auto')
    
    # Clean up numpy types
    analysis = convert_numpy(analysis)
    
    # Structure response for frontend
    return {
        'success': True,
        'type': 'video',
        'features': analysis,
        'summary': generate_video_summary(analysis)
    }


def generate_summary(features):
    """Generate human-readable summary from features"""
    
    if 'error' in features:
        return {
            'grade': 'N/A',
            'headline': 'Analysis Error',
            'description': features.get('error', 'Unknown error'),
            'key_insights': []
        }
    
    # Extract key metrics
    engagement = features.get('engagement_predictors', {})
    performance = features.get('predicted_performance', {})
    emotional = features.get('emotional_signals', {})
    
    # Calculate grade
    overall_score = performance.get('overall_effectiveness', 5)
    if overall_score >= 8:
        grade = 'A'
    elif overall_score >= 7:
        grade = 'B'
    elif overall_score >= 6:
        grade = 'C'
    elif overall_score >= 5:
        grade = 'D'
    else:
        grade = 'F'
    
    # Generate insights
    insights = []
    
    scroll_stop = engagement.get('scroll_stopping_power', 0)
    if scroll_stop >= 7:
        insights.append(f"✅ Strong scroll-stopping power ({scroll_stop}/10)")
    else:
        insights.append(f"⚠️ Weak scroll-stopping power ({scroll_stop}/10)")
    
    cta_present = features.get('call_to_action', {}).get('cta_present', False)
    if cta_present:
        insights.append("✅ Clear call-to-action present")
    else:
        insights.append("⚠️ No clear call-to-action")
    
    emotion = emotional.get('primary_emotion', 'unknown')
    intensity = emotional.get('emotional_intensity', 0)
    insights.append(f"😊 Primary emotion: {emotion} (intensity: {intensity}/10)")
    
    return {
        'grade': grade,
        'overall_score': overall_score,
        'headline': f'Grade {grade} - {get_grade_description(grade)}',
        'description': f'This ad scored {overall_score}/10 for overall effectiveness.',
        'key_insights': insights
    }


def generate_video_summary(analysis):
    """Generate summary for video analysis"""
    
    if 'error' in analysis:
        return {
            'grade': 'N/A',
            'headline': 'Analysis Error',
            'description': analysis.get('error', 'Unknown error'),
            'key_insights': []
        }
    
    vl = analysis.get('video_level_features', {})
    engagement = vl.get('overall_engagement', {})
    performance = vl.get('predicted_performance', {})
    
    # Calculate grade
    overall_score = performance.get('estimated_engagement', 5)
    if overall_score >= 8:
        grade = 'A'
    elif overall_score >= 7:
        grade = 'B'
    elif overall_score >= 6:
        grade = 'C'
    elif overall_score >= 5:
        grade = 'D'
    else:
        grade = 'F'
    
    # Generate insights
    insights = []
    
    hook_strength = engagement.get('first_3_seconds_hook', 0)
    if hook_strength >= 7:
        insights.append(f"✅ Strong opening hook ({hook_strength}/10)")
    else:
        insights.append(f"⚠️ Weak opening hook ({hook_strength}/10)")
    
    has_cta = engagement.get('has_clear_cta', False)
    if has_cta:
        insights.append("✅ Clear call-to-action present")
    else:
        insights.append("⚠️ No clear call-to-action")
    
    completion_rate = performance.get('estimated_completion_rate', 0)
    insights.append(f"📊 Estimated completion rate: {completion_rate}/10")
    
    return {
        'grade': grade,
        'overall_score': overall_score,
        'headline': f'Grade {grade} - {get_grade_description(grade)}',
        'description': f'This video ad scored {overall_score}/10 for engagement.',
        'key_insights': insights
    }


def get_grade_description(grade):
    descriptions = {
        'A': 'Excellent Performance',
        'B': 'Good Performance',
        'C': 'Average Performance',
        'D': 'Below Average',
        'F': 'Needs Improvement'
    }
    return descriptions.get(grade, 'Unknown')


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
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(results)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AD INTELLIGENCE API SERVER")
    print("="*70)
    print("\nStarting server on http://localhost:5001")
    print("\nEndpoints:")
    print("  POST /api/analyze - Upload and analyze ad")
    print("  GET  /api/results - List previous analyses")
    print("  GET  /api/health  - Health check")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)