"""
Video Ad Analyzer with Lava
Extracts key frames from video ads and analyzes them
"""

import os
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from lava_extractor import LavaAdFeatureExtractor
from datetime import datetime

class VideoAdAnalyzer:
    """
    Analyze video advertisements using Lava
    Extracts key frames and analyzes them
    """
    
    def __init__(self, model='gpt-4o-mini'):
        self.extractor = LavaAdFeatureExtractor(model=model, max_workers=1)
        self.model = model
    
    def extract_key_frames(self, video_path, num_frames='auto', method='uniform'):
        """
        Extract key frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract, or 'auto' to determine based on duration
            method: 'uniform' (evenly spaced) or 'smart' (scene changes)
        
        Returns:
            List of frame images (as numpy arrays)
        """
        
        print(f"🎬 Extracting frames from video...")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        # Auto-determine optimal number of frames based on duration
        if num_frames == 'auto':
            if duration <= 15:
                num_frames = 3  # Very short: 3 frames
            elif duration <= 30:
                num_frames = 5  # Short: 5 frames
            elif duration <= 60:
                num_frames = 8  # Medium: 8 frames
            elif duration <= 120:
                num_frames = 12  # Long: 12 frames
            else:
                num_frames = 20  # Very long: 20 frames
            
            print(f"   Auto-selected {num_frames} frames for {duration:.1f}s video")
        
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.1f}")
        print(f"   Extracting: {num_frames} key frames")
        
        frames = []
        
        if method == 'uniform':
            # Extract uniformly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append({
                        'frame': frame_rgb,
                        'timestamp': idx / fps,
                        'frame_number': idx
                    })
        
        elif method == 'smart':
            # Extract frames at scene changes (more advanced)
            prev_frame = None
            frame_count = 0
            scene_changes = []
            
            # Detect scene changes
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Calculate difference between frames
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_score = np.mean(diff)
                    
                    # If big change, it's a new scene
                    if diff_score > 30:  # Threshold
                        scene_changes.append(frame_count)
                
                prev_frame = frame
                frame_count += 1
            
            # Get frames at scene changes
            if scene_changes:
                selected_indices = scene_changes[:num_frames]
            else:
                # Fallback to uniform if no scene changes
                selected_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            for idx in selected_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append({
                        'frame': frame_rgb,
                        'timestamp': idx / fps,
                        'frame_number': idx
                    })
        
        cap.release()
        
        print(f"   ✅ Extracted {len(frames)} key frames")
        
        return frames
    
    def frame_to_base64(self, frame_data):
        """Convert numpy frame to base64"""
        from PIL import Image
        import io
        
        # Convert to PIL Image
        img = Image.fromarray(frame_data)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_video_frame(self, frame_base64, timestamp):
        """
        Analyze a single frame with temporal context
        """
        
        prompt = f"""Analyze this frame from a video advertisement (timestamp: {timestamp:.1f}s).

Extract features considering this is part of a video ad sequence.

Return ONLY valid JSON:

{{
  "temporal_context": {{
    "estimated_position": "opening/hook/middle/climax/cta/closing",
    "pacing": "fast/medium/slow",
    "momentum": 1-10
  }},
  "frame_content": {{
    "primary_focus": "product/person/text/scene/action",
    "visual_energy": 1-10,
    "information_density": 1-10
  }},
  "engagement_signals": {{
    "hook_strength": 1-10,
    "attention_retention": 1-10,
    "emotional_peak": 1-10,
    "cta_presence": true/false
  }},
  "video_specific": {{
    "motion_intensity": 1-10,
    "scene_complexity": 1-10,
    "text_legibility": 1-10,
    "sound_implied": "music/voice/silence/action/mixed"
  }}
}}"""
        
        response = self.extractor.call_lava_vision(frame_base64, prompt)
        
        # Parse JSON
        content = response['content']
        
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_str = content[json_start:json_end]
        
        return json.loads(json_str)
    
    def analyze_video(self, video_path, num_frames='auto'):
        """
        Complete video ad analysis
        
        Args:
            video_path: Path to video file
            num_frames: How many frames to analyze (or 'auto' to determine based on duration)
        
        Returns:
            Complete video analysis
        """
        
        print("\n" + "="*70)
        print("🎬 VIDEO AD ANALYSIS")
        print("="*70)
        print(f"\nVideo: {video_path}")
        print(f"Model: {self.model}")
        print(f"Frames to analyze: {num_frames}")
        print("="*70 + "\n")
        
        # Extract frames
        frames = self.extract_key_frames(video_path, num_frames=num_frames)
        
        # Analyze each frame
        print(f"\n🔄 Analyzing {len(frames)} frames...")
        
        frame_analyses = []
        
        for i, frame_data in enumerate(frames, 1):
            print(f"\n   [{i}/{len(frames)}] Analyzing frame at {frame_data['timestamp']:.1f}s...")
            
            # Convert to base64
            frame_base64 = self.frame_to_base64(frame_data['frame'])
            
            # Analyze
            analysis = self.analyze_video_frame(frame_base64, frame_data['timestamp'])
            analysis['timestamp'] = float(frame_data['timestamp'])  # Convert to Python float
            analysis['frame_number'] = int(frame_data['frame_number'])  # Convert to Python int
            
            frame_analyses.append(analysis)
        
        # Aggregate analysis
        print("\n📊 Aggregating video-level insights...")
        
        video_analysis = {
            'video_id': Path(video_path).stem,
            'frame_analyses': frame_analyses,
            'video_level_features': self._aggregate_features(frame_analyses),
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'model': self.model,
                'frames_analyzed': len(frames)
            }
        }
        
        print("\n✅ Video analysis complete!")
        
        return video_analysis
    
    def _aggregate_features(self, frame_analyses):
        """
        Aggregate frame-level features into video-level features
        """
        
        # Calculate averages and patterns
        avg_hook = np.mean([f['engagement_signals']['hook_strength'] for f in frame_analyses])
        avg_attention = np.mean([f['engagement_signals']['attention_retention'] for f in frame_analyses])
        avg_motion = np.mean([f['video_specific']['motion_intensity'] for f in frame_analyses])
        
        # Detect narrative structure
        has_opening_hook = frame_analyses[0]['engagement_signals']['hook_strength'] >= 7
        has_cta = any(f['engagement_signals']['cta_presence'] for f in frame_analyses)
        
        # Pacing analysis
        pacing_changes = sum(1 for i in range(1, len(frame_analyses)) 
                            if frame_analyses[i]['temporal_context']['pacing'] != 
                               frame_analyses[i-1]['temporal_context']['pacing'])
        
        return {
            'overall_engagement': {
                'average_hook_strength': round(avg_hook, 1),
                'average_attention_retention': round(avg_attention, 1),
                'first_3_seconds_hook': frame_analyses[0]['engagement_signals']['hook_strength'],
                'has_clear_cta': has_cta,
                'opening_hook_present': has_opening_hook
            },
            'video_dynamics': {
                'average_motion_intensity': round(avg_motion, 1),
                'pacing_variety': pacing_changes,
                'narrative_arc': self._detect_narrative_arc(frame_analyses)
            },
            'predicted_performance': {
                'estimated_completion_rate': self._estimate_completion_rate(frame_analyses),
                'estimated_engagement': round(avg_attention, 1),
                'scroll_stop_likelihood': frame_analyses[0]['engagement_signals']['hook_strength']
            }
        }
    
    def _detect_narrative_arc(self, frames):
        """Detect if video follows a narrative structure"""
        
        positions = [f['temporal_context']['estimated_position'] for f in frames]
        
        # Check for typical ad structure: hook → middle → cta
        has_structure = (
            'opening' in positions[0] or 'hook' in positions[0]
        ) and (
            'cta' in positions[-1] or 'closing' in positions[-1]
        )
        
        return 'clear' if has_structure else 'weak'
    
    def _estimate_completion_rate(self, frames):
        """Estimate video completion rate based on frame analysis"""
        
        # Strong first 3 seconds = better completion
        first_hook = frames[0]['engagement_signals']['hook_strength']
        avg_retention = np.mean([f['engagement_signals']['attention_retention'] for f in frames])
        
        # Simple formula
        score = (first_hook * 0.6 + avg_retention * 0.4)
        
        return round(score, 1)


def analyze_single_video(video_path, model='gpt-4o-mini', num_frames=5):
    """
    Quick function to analyze a single video
    """
    
    analyzer = VideoAdAnalyzer(model=model)
    analysis = analyzer.analyze_video(video_path, num_frames=num_frames)
    
    # Save results (convert numpy types to Python types for JSON)
    output_file = f"{Path(video_path).stem}_video_analysis.json"
    
    # Helper to convert numpy types
    def convert_numpy(obj):
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
    
    analysis_clean = convert_numpy(analysis)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_clean, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 VIDEO ANALYSIS SUMMARY")
    print("="*70)
    
    vl = analysis['video_level_features']
    
    print(f"\n🎯 Overall Engagement:")
    print(f"   First 3-sec hook: {vl['overall_engagement']['first_3_seconds_hook']}/10")
    print(f"   Average retention: {vl['overall_engagement']['average_attention_retention']}/10")
    print(f"   Has CTA: {'Yes' if vl['overall_engagement']['has_clear_cta'] else 'No'}")
    
    print(f"\n🎬 Video Dynamics:")
    print(f"   Motion intensity: {vl['video_dynamics']['average_motion_intensity']}/10")
    print(f"   Narrative arc: {vl['video_dynamics']['narrative_arc']}")
    
    print(f"\n📈 Predicted Performance:")
    print(f"   Completion rate: {vl['predicted_performance']['estimated_completion_rate']}/10")
    print(f"   Scroll-stop likelihood: {vl['predicted_performance']['scroll_stop_likelihood']}/10")
    
    print("\n" + "="*70)
    
    return analysis


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🎬 VIDEO AD ANALYZER")
        print("="*70)
        print("\nUsage: python video_analyzer.py <video_path> [model] [num_frames]")
        print("\nExamples:")
        print("  python video_analyzer.py ads/video001.mp4")
        print("  python video_analyzer.py ads/video001.mp4 gpt-4o-mini 10")
        print("  python video_analyzer.py ads/video001.mp4 gemini-pro 5")
        print("\nSupported formats: .mp4, .mov, .avi, .mkv")
        print("="*70)
    else:
        video_path = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else 'gpt-4o-mini'
        num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        analyze_single_video(video_path, model, num_frames)