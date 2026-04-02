
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from app import PronunciationApp

def verify_hybrid():
    app = PronunciationApp()
    audio_path = "data/school.wav"
    target_text = "학교"
    
    print(f"\n[Final Step 9 Verification: Hybrid Engine]")
    print("-" * 50)
    
    result = app.analyze_hybrid(audio_path, target_text)
    
    print(f"Hybrid Score: {result['total_score']:.1f}")
    print(f"Text Match: {result['is_match']}")
    print(f"Integrated Feedback:")
    for fb in result['feedback_details']:
        print(f"  - {fb}")
        
    # Test dialect acceptance (Simulation)
    print("\n[Testing Koryo-mar Dialect Acceptance...]")
    # 'ㅕ' -> 'ㅣ' or 'ㅔ' transition
    koryo_result = app.scorer.check_koryo_dialect_acceptance("ㅕ", 550.0, 1700.0) # 'ㅔ' formant values
    if koryo_result:
        print(f"Dialect Detected! Score: {koryo_result['score']:.1f}")
        print(f"Feedback: {koryo_result['feedback']}")

if __name__ == "__main__":
    verify_hybrid()
