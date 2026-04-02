
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from app import PronunciationApp

def run_scenarios():
    app = PronunciationApp()
    
    scenarios = [
        {"audio": "data/hello.wav", "target": "안녕하세요", "desc": "1. [정상 일치] '안녕하세요' 발음"},
        {"audio": "data/school.wav", "target": "안녕하세요", "desc": "2. [단어 불일치] '학교'라고 말하고 '안녕하세요'로 분석"},
        {"audio": "data/school.wav", "target": "학교", "desc": "3. [정밀 조음] '학교' 발음의 음소 정밀도 분석"}
    ]
    
    for scene in scenarios:
        print(f"\n{'='*20} {scene['desc']} {'='*20}")
        print(f"Audio: {scene['audio']}, Target: {scene['target']}")
        
        # 1. Acoustic Engine
        print("\n--- [Acoustic Engine 결과] ---")
        acoustic = app.analyze_pronunciation(scene['audio'], scene['target'])
        print(f"  총점: {acoustic['total_score']:.1f}점")
        print(f"  상세 피드백: {acoustic['feedback_details'][:2]} ...")
        
        # 2. WhisperX Engine
        print("\n--- [WhisperX Engine 결과] ---")
        whisper = app.analyze_with_whisperx(scene['audio'], scene['target'])
        print(f"  총점: {whisper['total_score']:.1f}점")
        print(f"  상세 피드백: {whisper['feedback_details']}")
        if whisper['analysis_raw']:
            print(f"  인식된 단어: {', '.join([w['word'] for w in whisper['analysis_raw']])}")

if __name__ == "__main__":
    run_scenarios()
