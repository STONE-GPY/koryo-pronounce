from src.g2p_engine import KoryoG2P
from src.audio_processor import AudioProcessor
from src.acoustic_analyzer import AcousticAnalyzer
from src.scorer import PronunciationScorer
import os

class PronunciationApp:
    """고려인 대상 발음 교정 시스템 통합 애플리케이션"""
    def __init__(self):
        self.g2p = KoryoG2P()
        self.audio_proc = AudioProcessor()
        self.analyzer = AcousticAnalyzer()
        self.scorer = PronunciationScorer()

    def analyze_pronunciation(self, audio_path: str, target_text: str) -> dict:
        """오디오와 제시된 문장을 비교하여 발음 분석 리포트 생성"""
        # 1. G2P 변환
        phonemes = self.g2p.convert(target_text)
        
        # 2. 오디오 전처리
        normalized_audio = self.audio_proc.load_and_normalize(audio_path)
        # 임시로 전처리된 오디오 저장 (분석용)
        temp_path = "data/temp_processed.wav"
        import soundfile as sf
        sf.write(temp_path, normalized_audio, 16000)
        
        # 3. 음향 분석 (예: 첫 번째 음절의 포먼트 및 기본 VOT 분석)
        # 실제 구현에서는 음절 분할이 선행되어야 함
        formants = self.analyzer.get_formants(temp_path)
        
        # 4. 스코어링 (예시로 첫 번째 모음 'ㅏ'가 포함된 경우 테스트)
        # 여기서는 구조적 시뮬레이션을 위해 고정된 타겟 음소 분석 수행
        vowel_score = self.scorer.score_vowel("ㅏ", formants["f1"], formants["f2"])
        
        # 5. 최종 리포트 구성
        report = {
            "target_text": target_text,
            "target_phonemes": phonemes,
            "total_score": vowel_score["score"],
            "feedback_details": [vowel_score["feedback"]],
            "analysis_raw": {
                "f1": formants["f1"],
                "f2": formants["f2"]
            }
        }
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return report

if __name__ == "__main__":
    # CLI 실행 예시
    import sys
    if len(sys.argv) < 3:
        print("Usage: python app.py <audio_path> <target_text>")
    else:
        app = PronunciationApp()
        result = app.analyze_pronunciation(sys.argv[1], sys.argv[2])
        import json
        print(json.dumps(result, indent=4, ensure_ascii=False))
