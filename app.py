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
        import uuid
        # 1. G2P 변환
        phonemes = self.g2p.convert(target_text)
        phonemes_no_space = phonemes.replace(" ", "")
        
        # 2. 오디오 전처리
        normalized_audio = self.audio_proc.load_and_normalize(audio_path)
        # 임시로 전처리된 오디오 저장 (분석용) - 동시성 에러(Race Condition) 방지
        temp_path = f"data/temp_processed_{uuid.uuid4().hex}.wav"
        import soundfile as sf
        from src.config import AudioConfig
        sf.write(temp_path, normalized_audio, AudioConfig.SAMPLE_RATE)
        
        # 3. 음향 분석 및 스코어링 (음절 동기화 및 이중모음 지원)
        # 화자 Pitch(F0) 측정 (개인화 스케일링용)
        user_pitch = self.analyzer.get_pitch(temp_path)
        
        import jamo
        import librosa
        
        duration = librosa.get_duration(path=temp_path) # filename -> path (DeprecationWarning 해결)
        num_syllables = len(phonemes_no_space) # 공백 제외 순수 발음 음절 수
        segment_duration = duration / num_syllables if num_syllables > 0 else duration
        
        scores = []
        feedback_list = []
        raw_analysis = []
        
        vowel_types = []
        syllable_vowels = []
        syllable_plosives = []
        
        for char in phonemes_no_space:
            decomposed = jamo.j2hcj(jamo.h2j(char))
            
            # 모음 탐색 (이중모음 우선)
            v_type = None
            v_char = None
            for p in decomposed:
                if p in self.scorer.diphthong_standards:
                    v_type = "diphthong"
                    v_char = p
                    break
                elif p in self.scorer.vowel_standards:
                    v_type = "monophthong"
                    v_char = p
                    break
            
            vowel_types.append(v_type if v_type else "monophthong")
            syllable_vowels.append(v_char)
            
            # 파열음 탐색
            p_char = None
            for p in decomposed:
                if p in self.scorer.vot_standards:
                    p_char = p
                    break
            syllable_plosives.append(p_char)
            
        segments_formants = self.analyzer.get_formants_for_segments(temp_path, vowel_types)
        
        for i in range(num_syllables):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            # 파열음 스코어링 (음절 단위 독립 매칭)
            p_char = syllable_plosives[i]
            if p_char:
                vot_ms = self.analyzer.estimate_plosive_vot(temp_path, start_time, end_time)
                plosive_score = self.scorer.score_plosive(p_char, vot_ms)
                scores.append(plosive_score["score"])
                feedback_list.append(plosive_score["feedback"])
                raw_analysis.append({"phoneme": p_char, "vot_ms": vot_ms, "segment": i+1})
                
            # 모음 스코어링 (이중/단모음 구분)
            v_char = syllable_vowels[i]
            v_type = vowel_types[i]
            if v_char and i < len(segments_formants):
                seg_data = segments_formants[i]
                if v_type == "diphthong":
                    score_res = self.scorer.score_diphthong(v_char, seg_data["start_f1"], seg_data["start_f2"], seg_data["end_f1"], seg_data["end_f2"], user_pitch)
                    scores.append(score_res["score"])
                    feedback_list.append(score_res["feedback"])
                    raw_analysis.append({"phoneme": v_char, "start_f1": seg_data["start_f1"], "end_f1": seg_data["end_f1"], "segment": i+1})
                else:
                    score_res = self.scorer.score_vowel(v_char, seg_data["f1"], seg_data["f2"], user_pitch)
                    scores.append(score_res["score"])
                    feedback_list.append(score_res["feedback"])
                    raw_analysis.append({"phoneme": v_char, "f1": seg_data["f1"], "f2": seg_data["f2"], "segment": i+1})
            
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
