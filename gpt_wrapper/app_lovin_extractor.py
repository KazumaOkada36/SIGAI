"""
AppLovin Ad Intelligence Feature Extractor
Images & Videos (with audio features + optional transcript)
"""

import os, io, time, base64, json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import librosa
from PIL import Image
from tqdm import tqdm
import subprocess
import imageio_ffmpeg
from openai import OpenAI

USE_MOCK = os.getenv("MOCK_EXTRACTOR", "0") == "1"
# -----------------------
# Helpers (deduplicated)
# -----------------------
import random

def _fake_features_for(path: Path, is_video: bool) -> dict:
    rng = random.Random(hash(path.stem) & 0xffffffff)
    def r1_10(): return rng.randint(1, 10)
    def rbool():  return rng.choice([True, False])
    def pick(*xs): return rng.choice(xs)

    feat = {
      "emotional_signals": {
        "primary_emotion": pick("joy","excitement","trust","urgency","curiosity"),
        "emotional_intensity": r1_10(), "aspirational_appeal": r1_10(),
        "humor_present": rbool(), "creates_fomo": rbool(),
        "evokes_nostalgia": rbool(), "trust_building_elements": r1_10()
      },
      "visual_composition": {
        "visual_complexity": r1_10(), "color_scheme": pick("vibrant","muted","bold","pastel"),
        "dominant_colors": ["red","blue"], "contrast_level": r1_10(),
        "whitespace_usage": r1_10(), "symmetry_score": r1_10(),
        "motion_energy": r1_10(), "professional_polish": r1_10()
      },
      "human_elements": {
        "people_present": rbool(), "number_of_faces": rng.randint(0,3),
        "facial_expressions": pick("happy","neutral","varied"),
        "age_demographic_shown": pick("young_adults","mixed","none"),
        "gender_representation": pick("male","female","mixed","none"),
        "diversity_shown": rbool(), "celebrity_or_influencer": rbool(),
        "relatable_characters": r1_10()
      },
      "text_and_messaging": {
        "text_density": r1_10(), "headline_present": rbool(),
        "headline_text": "Sale today" if rbool() else None,
        "headline_word_count": rng.randint(0,8), "subheading_present": rbool(),
        "readability_score": r1_10(), "power_words_count": rng.randint(0,5),
        "benefit_focused": rbool(), "problem_solution_framing": rbool()
      },
      "call_to_action": {
        "cta_present": rbool(), "cta_text": pick("Buy now","Learn more",None),
        "cta_prominence": r1_10(), "cta_action_verb": pick("buy","learn","get","join","none"),
        "cta_urgency": r1_10(), "cta_friction": r1_10()
      },
      "product_content": {
        "product_category": pick("app","ecommerce","service","tech","gaming","other"),
        "product_visible": rbool(), "product_in_use": rbool(),
        "product_benefits_shown": r1_10(), "before_after_present": rbool(),
        "price_shown": rbool(), "discount_or_offer": rbool(), "limited_time_offer": rbool()
      },
      "branding": {
        "logo_present": rbool(), "logo_prominence": r1_10(),
        "logo_placement": pick("top_left","top_right","center","bottom_left","bottom_right","none"),
        "brand_name_visible": rbool(), "brand_personality": pick("playful","serious","friendly","innovative")
      },
      "engagement_predictors": {
        "scroll_stopping_power": r1_10(), "first_3_sec_hook": r1_10(),
        "curiosity_gap": r1_10(), "social_proof_elements": r1_10(),
        "scarcity_indicators": r1_10(), "pattern_interruption": r1_10(), "memability": r1_10()
      },
      "technical_quality": {
        "image_quality": r1_10(), "mobile_optimized": r1_10(),
        "text_legibility": r1_10(), "load_speed_friendly": r1_10(),
        "aspect_ratio": pick("square","vertical","horizontal")
      },
      "content_type": {
        "lifestyle_imagery": rbool(), "product_showcase": rbool(),
        "testimonial_style": rbool(), "explainer_format": rbool(),
        "meme_style": rbool(), "cinematic_style": rbool(), "ugc_style": rbool()
      },
      "psychological_triggers": {
        "authority_signals": r1_10(), "reciprocity_elements": r1_10(),
        "social_validation": r1_10(), "commitment_consistency": r1_10(),
        "liking_similarity": r1_10(), "scarcity_urgency": r1_10()
      },
      "predicted_performance": {
        "estimated_ctr": r1_10(), "estimated_engagement": r1_10(),
        "estimated_conversion_potential": r1_10(), "virality_potential": r1_10(),
        "overall_effectiveness": r1_10()
      },
      "audio_analysis": {
        "has_audio": is_video, "tempo_bpm": 110.0 if is_video else 0.0,
        "rms_mean": 0.05, "rms_std": 0.01, "spec_centroid_mean": 1800.0,
        "spec_bw_mean": 1200.0, "zcr_mean": 0.05, "flatness_mean": 0.2,
        "mfcc_means": [0.0]*13, "mfcc_stds": [1.0]*13, "transcript": "" }
    }
    feat["_meta"] = {
        "ad_id": path.stem, "extraction_timestamp": datetime.utcnow().isoformat(),
        "model": "mock", "media_type": "video" if is_video else "image",
        "frames_used": 8 if is_video else 1, "source_path": str(path)
    }
    return feat

def _is_image_path(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}

def _is_video_path(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}

def _sample_video_frames(path: str, k: int = 8):
    """Uniformly sample k frames from a video; return list of RGB arrays."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total <= 0:
        idxs = list(range(k))
    elif total <= k:
        idxs = list(range(total))
        while len(idxs) < k:
            idxs += idxs[:k - len(idxs)]
        idxs = idxs[:k]
    else:
        step = total / k
        idxs = [int(i * step + step / 2) for i in range(k)]

    frames = []
    for t in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read: {path}")

    while len(frames) < k:
        frames.append(frames[-1])
    return frames[:k]

def _extract_audio_array(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Extract mono float32 PCM audio using ffmpeg (via imageio-ffmpeg).
    Returns an empty array if no audio.
    """
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-v", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_bytes = proc.stdout
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        return np.frombuffer(audio_bytes, dtype=np.float32)
    except subprocess.CalledProcessError:
        return np.array([], dtype=np.float32)

def _audio_feature_vector(y: np.ndarray, sr: int = 16000) -> dict:
    """Compact audio features for tone."""
    if y.size == 0:
        return {
            "has_audio": False, "tempo_bpm": 0.0,
            "rms_mean": 0.0, "rms_std": 0.0,
            "spec_centroid_mean": 0.0, "spec_bw_mean": 0.0,
            "zcr_mean": 0.0, "flatness_mean": 0.0,
            "mfcc_means": [0.0]*13, "mfcc_stds": [0.0]*13
        }
    if y.size < sr:
        y = np.pad(y, (0, sr - y.size))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms  = librosa.feature.rms(y=y)[0]
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sbw  = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr  = librosa.feature.zero_crossing_rate(y=y)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr, aggregate=None)
        tempo = float(np.median(tempo)) if tempo.size else 0.0
    except Exception:
        tempo = 0.0

    return {
        "has_audio": True, "tempo_bpm": tempo,
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
        "spec_centroid_mean": float(sc.mean()), "spec_bw_mean": float(sbw.mean()),
        "zcr_mean": float(zcr.mean()), "flatness_mean": float(flat.mean()),
        "mfcc_means": [float(x) for x in mfcc.mean(axis=1)],
        "mfcc_stds":  [float(x) for x in mfcc.std(axis=1)],
    }

def _summarize_audio_for_prompt(aud: dict) -> str:
    if not aud.get("has_audio", False):
        return "No audio track detected."
    energy = "high" if aud["rms_mean"] > 0.1 else ("medium" if aud["rms_mean"] > 0.03 else "low")
    pace   = "fast" if aud["tempo_bpm"] >= 120 else ("moderate" if aud["tempo_bpm"] >= 90 else "slow")
    brightness = "bright" if aud["spec_centroid_mean"] > 2500 else ("neutral" if aud["spec_centroid_mean"] > 1500 else "warm")
    return (f"Audio cues → energy: {energy}; tempo: {aud['tempo_bpm']:.0f} BPM ({pace}); "
            f"tone color: {brightness}; zcr: {aud['zcr_mean']:.3f}; flatness: {aud['flatness_mean']:.3f}.")

def _openai_transcribe_whisper(client: OpenAI, path: str, target_sr: int = 16000) -> str:
    """Optional: transcribe VO with Whisper; returns plain text or ''."""
    try:
        import soundfile as sf
        tmp_wav = Path(path).with_suffix(".tmp.wav")
        y = _extract_audio_array(path, target_sr)
        if y.size == 0:
            return ""
        sf.write(str(tmp_wav), y, target_sr)
        with open(tmp_wav, "rb") as f:
            tx = client.audio.transcriptions.create(model="whisper-1", file=f)
        try:
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            pass
        return getattr(tx, "text", "") or ""
    except Exception:
        return ""


# -----------------------
# Extractor
# -----------------------
class AdFeatureExtractor:
    """
    Extract high-value features from ad creatives using GPT-4 Vision
    Images + Videos (frames + audio summary + optional transcript)
    """
    def __init__(self, api_key: str | None = None, max_workers: int = 10):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required!")
        self.client = OpenAI(api_key=api_key)
        self.max_workers = max_workers

    def extract_features(self, media_path: str, video_frames: int = 8, include_transcript: bool = True) -> dict:
        """
        Image OR video:
          - Images: send the image
          - Videos: send k frames + audio summary (+ transcript if enabled)
        Returns dict with your schema + 'audio_analysis' section.
        """
        p = Path(media_path)
        if USE_MOCK:
            return _fake_features_for(p, is_video=_is_video_path(p))
        # === Main prompt (your schema) ===
        base_prompt = """You are an expert ad intelligence system analyzing creatives for a recommendation engine.

Extract EVERY possible signal from this advertisement that could predict engagement, clickthrough, or conversion.

**Return ONLY valid JSON in this exact structure (no markdown, no explanation):**

{
  "emotional_signals": {
    "primary_emotion": "joy/excitement/trust/fear/urgency/calm/curiosity/desire/nostalgia",
    "emotional_intensity": 1-10,
    "aspirational_appeal": 1-10,
    "humor_present": true/false,
    "creates_fomo": true/false,
    "evokes_nostalgia": true/false,
    "trust_building_elements": 1-10
  },
  "visual_composition": {
    "visual_complexity": 1-10,
    "color_scheme": "vibrant/muted/monochrome/pastel/bold/earthy",
    "dominant_colors": ["color1", "color2"],
    "contrast_level": 1-10,
    "whitespace_usage": 1-10,
    "symmetry_score": 1-10,
    "professional_polish": 1-10
  },
  "human_elements": {
    "people_present": true/false,
    "number_of_faces": 0-10,
    "facial_expressions": "happy/serious/surprised/neutral/varied",
    "age_demographic_shown": "children/teens/young_adults/middle_aged/seniors/mixed",
    "gender_representation": "male/female/mixed/none",
    "diversity_shown": true/false,
    "celebrity_or_influencer": true/false,
    "relatable_characters": 1-10
  },
  "text_and_messaging": {
    "text_density": 1-10,
    "headline_present": true/false,
    "headline_text": "exact text or null",
    "headline_word_count": 0-20,
    "subheading_present": true/false,
    "readability_score": 1-10,
    "power_words_count": 0-10,
    "benefit_focused": true/false,
    "problem_solution_framing": true/false
  },
  "call_to_action": {
    "cta_present": true/false,
    "cta_text": "exact text or null",
    "cta_prominence": 1-10,
    "cta_action_verb": "download/buy/learn/sign_up/try/get/join/other/none",
    "cta_urgency": 1-10,
    "cta_friction": 1-10
  },
  "product_content": {
    "product_category": "app/ecommerce/service/brand_awareness/automotive/fashion/food/finance/health/tech/gaming/other",
    "product_visible": true/false,
    "product_in_use": true/false,
    "product_benefits_shown": 1-10,
    "before_after_present": true/false,
    "price_shown": true/false,
    "discount_or_offer": true/false,
    "limited_time_offer": true/false
  },
  "branding": {
    "logo_present": true/false,
    "logo_prominence": 1-10,
    "logo_placement": "top_left/top_right/center/bottom_left/bottom_right/none",
    "brand_name_visible": true/false,
    "brand_personality": "playful/serious/luxury/friendly/innovative/trustworthy/edgy/other"
  },
  "engagement_predictors": {
    "scroll_stopping_power": 1-10,
    "first_3_sec_hook": 1-10,
    "curiosity_gap": 1-10,
    "social_proof_elements": 1-10,
    "scarcity_indicators": 1-10,
    "pattern_interruption": 1-10,
    "memability": 1-10
  },
  "technical_quality": {
    "image_quality": 1-10,
    "mobile_optimized": 1-10,
    "text_legibility": 1-10,
    "load_speed_friendly": 1-10,
    "aspect_ratio": "square/vertical/horizontal/other"
  },
  "content_type": {
    "lifestyle_imagery": true/false,
    "product_showcase": true/false,
    "testimonial_style": true/false,
    "explainer_format": true/false,
    "meme_style": true/false,
    "cinematic_style": true/false,
    "ugc_style": true/false
  },
  "psychological_triggers": {
    "authority_signals": 1-10,
    "reciprocity_elements": 1-10,
    "social_validation": 1-10,
    "commitment_consistency": 1-10,
    "liking_similarity": 1-10,
    "scarcity_urgency": 1-10
  },
  "predicted_performance": {
    "estimated_ctr": 1-10,
    "estimated_engagement": 1-10,
    "estimated_conversion_potential": 1-10,
    "virality_potential": 1-10,
    "overall_effectiveness": 1-10
  }
}

Rate ALL numeric fields on appropriate 1-10 scales. Be precise and analytical. Base everything on what you actually see/hear from the ad.
"""

        content_items = []

        # === If video, compute audio features (+ optional transcript) and prepend as text ===
        is_video = _is_video_path(p)
        audio_feats = {}
        transcript = ""
        if is_video:
            try:
                y = _extract_audio_array(str(p), target_sr=16000)
                audio_feats = _audio_feature_vector(y, sr=16000)
                audio_summary = _summarize_audio_for_prompt(audio_feats)
                if include_transcript:
                    transcript = _openai_transcribe_whisper(self.client, str(p), target_sr=16000) or ""
                block = "AUDIO SUMMARY: " + audio_summary
                if transcript:
                    t_short = transcript.strip()
                    if len(t_short) > 1200:
                        t_short = t_short[:1200] + "…"
                    block += "\nTRANSCRIPT (truncated): " + t_short
                prompt_text = block + "\n\n" + base_prompt
            except Exception as e:
                prompt_text = f"AUDIO SUMMARY: Audio extraction failed: {e}\n\n" + base_prompt
                audio_feats = {"has_audio": False}
        else:
            prompt_text = base_prompt

        content_items.append({"type": "text", "text": prompt_text})

        # === Add visual inputs ===
        if _is_image_path(p):
            with open(p, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}
            })
        elif is_video:
            frames = _sample_video_frames(str(p), k=video_frames)
            for fr in frames:
                buf = io.BytesIO()
                Image.fromarray(fr).save(buf, format="JPEG", quality=90)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                })
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        # === Call OpenAI Vision ===
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": content_items}],
                max_tokens=2000,
                temperature=0.1
            )
            content = resp.choices[0].message.content

            # Extract JSON from the model response
            if '```json' in content:
                s = content.find('```json') + 7
                e = content.find('```', s)
                json_str = content[s:e].strip()
            elif '```' in content:
                s = content.find('```') + 3
                e = content.find('```', s)
                json_str = content[s:e].strip()
            else:
                s = content.find('{'); e = content.rfind('}') + 1
                json_str = content[s:e]
            features = json.loads(json_str)

            # Attach metadata + audio_analysis
            features["_meta"] = {
                "ad_id": p.stem,
                "extraction_timestamp": datetime.now().isoformat(),
                "model": "gpt-4-vision-preview",
                "media_type": "video" if is_video else "image",
                "frames_used": video_frames if is_video else 1,
                "source_path": str(p)
            }
            features["audio_analysis"] = {
                "has_audio": audio_feats.get("has_audio", False),
                "tempo_bpm": audio_feats.get("tempo_bpm", 0.0),
                "rms_mean": audio_feats.get("rms_mean", 0.0),
                "rms_std": audio_feats.get("rms_std", 0.0),
                "spec_centroid_mean": audio_feats.get("spec_centroid_mean", 0.0),
                "spec_bw_mean": audio_feats.get("spec_bw_mean", 0.0),
                "zcr_mean": audio_feats.get("zcr_mean", 0.0),
                "flatness_mean": audio_feats.get("flatness_mean", 0.0),
                "mfcc_means": audio_feats.get("mfcc_means", [0.0]*13),
                "mfcc_stds": audio_feats.get("mfcc_stds", [0.0]*13),
                "transcript": transcript if is_video and include_transcript else ""
            }
            return features

        except Exception as e:
            print(f"❌ Error extracting features from {p}: {e}")
            return {
                "_meta": {
                    "ad_id": p.stem,
                    "error": str(e),
                    "media_type": "video" if is_video else "image",
                    "source_path": str(p)
                },
                "audio_analysis": {"has_audio": False}
            }

    def process_single_ad(self, path_str: str):
        """Process single ad with error handling."""
        try:
            return path_str, self.extract_features(path_str)
        except Exception as e:
            return path_str, {"_meta": {"ad_id": Path(path_str).stem, "error": str(e)}}

    def process_dataset(self, input_dir: str, output_file: str = "applovin_features.json"):
        """Process entire dataset (images + videos) in parallel."""
        print("\n" + "="*70)
        print("🚀 AD INTELLIGENCE FEATURE EXTRACTOR")
        print("="*70)
        print(f"\nInput directory: {input_dir}")
        print(f"Output file: {output_file}")
        print(f"Parallel workers: {self.max_workers}")
        print("="*70 + "\n")

        media_files: list[Path] = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.mp4', '*.mov', '*.mkv', '*.avi']:
            media_files.extend(Path(input_dir).rglob(ext))
        print(f"🎞️ Found {len(media_files)} media files (images/videos)")
        if not media_files:
            print("❌ No media found!")
            return None

        # Rough estimate (videos send multiple frames)
        estimated_images = sum(1 if _is_image_path(p) else 8 for p in media_files)  # assume 8 frames/video
        print(f"⏱️  Rough calls worth of images: ~{estimated_images}")

        proceed = input("\n▶️  Proceed? (yes/no): ").strip().lower()
        if proceed not in {"yes", "y"}:
            print("Cancelled.")
            return None

        results = {}
        start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex, tqdm(total=len(media_files), desc="Extracting") as bar:
            futs = {ex.submit(self.process_single_ad, str(p)): p for p in media_files}
            for fut in as_completed(futs):
                path_str, feats = fut.result()
                ad_id = Path(path_str).stem
                results[ad_id] = feats
                bar.update(1)

        elapsed = time.time() - start
        print(f"\n✅ Processed {len(results)} ads in {elapsed:.1f}s (avg {elapsed/max(1,len(results)):.2f}s/ad)")

        # Save JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Flatten to CSV
        df = self.features_to_dataframe(results)
        csv_file = output_file.replace(".json", ".csv")
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved\n- {output_file}\n- {csv_file}")
        self.print_summary(df)
        return df

    @staticmethod
    def features_to_dataframe(results: dict) -> pd.DataFrame:
        """Flatten nested JSON to DataFrame."""
        rows = []
        for ad_id, features in results.items():
            if not isinstance(features, dict):
                continue
            if "error" in features.get("_meta", {}):
                continue
            row = {"ad_id": ad_id}
            for category, values in features.items():
                if category == "_meta":
                    continue
                if isinstance(values, dict):
                    for k, v in values.items():
                        row[f"{category}_{k}"] = v
                else:
                    row[category] = values
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def print_summary(df: pd.DataFrame):
        """Print quick summary stats."""
        print("\n" + "="*70)
        print("📊 FEATURE EXTRACTION SUMMARY")
        print("="*70)
        print(f"\n📈 Dataset size: {len(df)} ads")
        print(f"📊 Total features extracted: {len(df.columns)} columns")
        if 'engagement_predictors_scroll_stopping_power' in df.columns:
            m = df['engagement_predictors_scroll_stopping_power'].mean()
            print(f"🎯 Avg scroll-stopping power: {m:.1f}/10")
        if 'predicted_performance_overall_effectiveness' in df.columns:
            m = df['predicted_performance_overall_effectiveness'].mean()
            print(f"🎯 Avg overall effectiveness: {m:.1f}/10")
        if 'call_to_action_cta_present' in df.columns:
            cta = df['call_to_action_cta_present'].sum() / max(1, len(df)) * 100
            print(f"🎯 Ads with CTA: {cta:.1f}%")
        if 'human_elements_people_present' in df.columns:
            ppl = df['human_elements_people_present'].sum() / max(1, len(df)) * 100
            print(f"🎯 Ads with people: {ppl:.1f}%")
        print("="*70)


def main():
    import sys
    if len(sys.argv) < 2:
        print("\nUsage: python applovin_extractor.py <ads_directory>\n")
        return
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        print("\n💡 Set it with:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        return



    ads_dir = sys.argv[1]
    extractor = AdFeatureExtractor(max_workers=10)
    extractor.process_dataset(ads_dir, output_file="applovin_features.json")


if __name__ == "__main__":
    main()
