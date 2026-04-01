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
        
        # 3. 음향 분석 및 스코어링 (다중 모음 지원)
        # 화자 Pitch(F0) 측정 (개인화 스케일링용)
        user_pitch = self.analyzer.get_pitch(temp_path)
        
        import jamo
        decomposed = jamo.j2hcj(jamo.h2j(phonemes))
        target_vowels = [p for p in decomposed if p in self.scorer.vowel_standards]
        target_plosives = [p for p in decomposed if p in self.scorer.vot_standards]
        
        scores = []
        feedback_list = []
        raw_analysis = []
        
        # 파열음(VOT) 스코어링
        if len(target_plosives) > 0:
            vot_ms = self.analyzer.estimate_plosive_vot(temp_path)
            # 여기서는 오디오 전체에서 가장 뚜렷한 onset의 VOT를 모든 파열음 평가에 일괄 적용 (단순화)
            for plosive in target_plosives:
                plosive_score = self.scorer.score_plosive(plosive, vot_ms)
                scores.append(plosive_score["score"])
                feedback_list.append(plosive_score["feedback"])
                raw_analysis.append({"phoneme": plosive, "vot_ms": vot_ms})
                
        # 모음(Formant) 스코어링
        if len(target_vowels) > 0:
            segments_formants = self.analyzer.get_formants_for_segments(temp_path, len(target_vowels))
            for i, vowel in enumerate(target_vowels):
                f1 = segments_formants[i]["f1"]
                f2 = segments_formants[i]["f2"]
                vowel_score = self.scorer.score_vowel(vowel, f1, f2, user_pitch=user_pitch)
                
                scores.append(vowel_score["score"])
                feedback_list.append(vowel_score["feedback"])
                raw_analysis.append({"phoneme": vowel, "f1": f1, "f2": f2, "time": segments_formants[i]["time"]})
            
        if len(scores) > 0:
            total_score = sum(scores) / len(scores)
        else:
            total_score = 100
            feedback_list.append("분석할 수 있는 음소가 없습니다.")
        
        # 5. 최종 리포트 구성
        report = {
            "target_text": target_text,
            "target_phonemes": phonemes,
            "total_score": total_score,
            "feedback_details": feedback_list,
            "analysis_raw": raw_analysis
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
