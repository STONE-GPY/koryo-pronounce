
import sys
import os
import torch

# Add src to path
sys.path.append(os.getcwd())

from app import PronunciationApp

def compare_engines(audio_path, target_text):
    app = PronunciationApp()
    
    print(f"\n[Target: {target_text}]")
    print("-" * 50)
    
    # 1. Acoustic Engine (Current Main)
    print("1. [Acoustic Engine] Analyzing...")
    acoustic_report = app.analyze_pronunciation(audio_path, target_text)
    print(f"   - Score: {acoustic_report['total_score']:.1f}")
    print(f"   - Feedback: {acoustic_report['feedback_details'][:2]} ...")
    
    # 2. WhisperX Engine (Jules Branch)
    print("\n2. [WhisperX Engine] Analyzing...")
    try:
        whisper_report = app.analyze_with_whisperx(audio_path, target_text)
        print(f"   - Score: {whisper_report['total_score']:.1f}")
        print(f"   - Feedback: {whisper_report['feedback_details']}")
    except Exception as e:
        print(f"   - WhisperX failed: {e}")

    # 3. Evolution: Hybrid Concept (Step 4 & 5)
    print("\n3. [Evolved Hybrid Concept] (Simulated)")
    # Logic: Use WhisperX for word segmentation, then Acoustic for phoneme precision
    hybrid_score = (acoustic_report['total_score'] * 0.4) + (whisper_report['total_score'] * 0.6)
    print(f"   - Hybrid Score: {hybrid_score:.1f}")
    combined_feedback = []
    if whisper_report['total_score'] > 90:
        combined_feedback.append("Excellent recognition!")
        combined_feedback.extend(acoustic_report['feedback_details'][:2])
    else:
        combined_feedback.append("Word clarity needs improvement.")
        
    print(f"   - Integrated Feedback: {combined_feedback}")

if __name__ == "__main__":
    audio_path = "data/sample.wav"
    target_text = "안녕하세요"
    
    if os.path.exists(audio_path):
        compare_engines(audio_path, target_text)
    else:
        print(f"File not found: {audio_path}")
