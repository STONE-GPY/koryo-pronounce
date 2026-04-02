
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from app import PronunciationApp

def test_dual_engine():
    app = PronunciationApp()
    audio_path = "data/sample.wav"
    target_text = "학교" # We found sample.wav is actually "학교"
    
    print(f"Testing Dual Engine with: {target_text}")
    
    # Test Acoustic
    print("\n[Testing Acoustic Engine...]")
    acoustic = app.analyze_pronunciation(audio_path, target_text)
    print(f"Score: {acoustic['total_score']:.1f}")
    print(f"Feedback: {acoustic['feedback_details']}")

    # Test WhisperX
    print("\n[Testing WhisperX Engine...]")
    whisper = app.analyze_with_whisperx(audio_path, target_text)
    print(f"Score: {whisper['total_score']:.1f}")
    print(f"Feedback: {whisper['feedback_details']}")
    if whisper['analysis_raw']:
        print(f"Detected Word: {whisper['analysis_raw'][0]['word']} (Conf: {whisper['analysis_raw'][0]['confidence']:.2f})")

if __name__ == "__main__":
    test_dual_engine()
