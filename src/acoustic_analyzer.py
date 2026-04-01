import parselmouth
import numpy as np

class AcousticAnalyzer:
    """음성학적 분석 (Formant, VOT) 담당"""
    def __init__(self):
        pass

    def get_formants(self, audio_file: str) -> dict:
        """Parselmouth (Praat)를 사용하여 F1, F2 추출"""
        sound = parselmouth.Sound(audio_file)
        # Praat Formant 분석 설정
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, 
                                        maximum_formant=5500.0, window_length=0.025, pre_emphasis_from=50.0)
        
        # 중간 지점의 포먼트 값 추출
        mid_time = sound.get_total_duration() / 2
        f1 = formant.get_value_at_time(1, mid_time)
        f2 = formant.get_value_at_time(2, mid_time)
        
        return {
            "f1": f1 if not np.isnan(f1) else 0.0,
            "f2": f2 if not np.isnan(f2) else 0.0
        }

    def measure_vot(self, audio_file: str, burst_time: float, voicing_onset: float) -> float:
        """VOT (Voice Onset Time) 계산 (ms 단위)"""
        # VOT = (진동 시작 시간 - 파열음 버스트 시간) * 1000
        vot_ms = (voicing_onset - burst_time) * 1000
        return vot_ms

    def get_pitch(self, audio_file: str) -> float:
        """Parselmouth를 사용하여 평균 Pitch(F0) 추출"""
        sound = parselmouth.Sound(audio_file)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0] # 무음/unvoiced 구간 제외
        if len(pitch_values) > 0:
            return float(np.mean(pitch_values))
        return 0.0

    def estimate_gender(self, audio_file: str) -> str:
        """Pitch를 기반으로 성별 추정 (165Hz를 임계값으로 사용)"""
        mean_pitch = self.get_pitch(audio_file)
        if mean_pitch == 0.0:
            return "unknown"
        return "female" if mean_pitch > 165 else "male"

    def analyze_vowel_space(self, f1: float, f2: float) -> str:
        """포먼트 기반 모음 위치 판별 (간단한 로직)"""
        # 'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ'의 대략적인 F1/F2 범위 비교
        # 추후 상세 데이터 기반 룩업 테이블 적용 가능
        return "Not Implemented"
