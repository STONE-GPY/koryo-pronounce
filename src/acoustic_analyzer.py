import parselmouth
import numpy as np

class AcousticAnalyzer:
    """음성학적 분석 (Formant, VOT) 담당"""
    def __init__(self):
        pass

    def get_formants(self, audio_file: str) -> dict:
        """Parselmouth (Praat)를 사용하여 최고 강도(Intensity) 지점의 F1, F2 추출"""
        sound = parselmouth.Sound(audio_file)
        # Praat Formant 분석 설정
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, 
                                        maximum_formant=5500.0, window_length=0.025, pre_emphasis_from=50.0)
        
        # 최고 에너지(Intensity) 지점 탐색 (모음의 핵심 발음 구간일 확률이 높음)
        try:
            intensity = sound.to_intensity()
            max_frame_idx = np.argmax(intensity.values[0])
            target_time = intensity.get_time_from_frame_number(max_frame_idx + 1)
        except Exception:
            target_time = sound.get_total_duration() / 2
        
        f1 = formant.get_value_at_time(1, target_time)
        f2 = formant.get_value_at_time(2, target_time)
        
        return {
            "f1": float(f1) if not np.isnan(f1) else 0.0,
            "f2": float(f2) if not np.isnan(f2) else 0.0,
            "time": float(target_time)
        }

    def get_formants_for_segments(self, audio_file: str, vowel_types: list) -> list:
        """오디오를 음절 개수만큼 구간으로 나누고, 모음 종류에 따라 포먼트 추출
        vowel_types: 각 구간의 모음 타입 (단모음이면 'monophthong', 이중모음이면 'diphthong')"""
        num_segments = len(vowel_types)
        if num_segments <= 0:
            return []
            
        sound = parselmouth.Sound(audio_file)
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, 
                                        maximum_formant=5500.0, window_length=0.025, pre_emphasis_from=50.0)
        duration = sound.get_total_duration()
        results = []
        
        try:
            intensity = sound.to_intensity()
            times = intensity.xs()
            segment_duration = duration / num_segments
            
            for i, v_type in enumerate(vowel_types):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                valid_indices = np.where((times >= start_time) & (times < end_time))[0]
                
                if v_type == 'diphthong':
                    # 이중모음은 구간의 20% 지점과 80% 지점에서 추출
                    t_start = start_time + (segment_duration * 0.2)
                    t_end = start_time + (segment_duration * 0.8)
                    f1_start = formant.get_value_at_time(1, t_start)
                    f2_start = formant.get_value_at_time(2, t_start)
                    f1_end = formant.get_value_at_time(1, t_end)
                    f2_end = formant.get_value_at_time(2, t_end)
                    results.append({
                        "type": "diphthong",
                        "start_f1": float(f1_start) if not np.isnan(f1_start) else 0.0,
                        "start_f2": float(f2_start) if not np.isnan(f2_start) else 0.0,
                        "end_f1": float(f1_end) if not np.isnan(f1_end) else 0.0,
                        "end_f2": float(f2_end) if not np.isnan(f2_end) else 0.0,
                        "time": float(start_time + segment_duration/2)
                    })
                else:
                    # 단모음은 최고 강도 지점에서 추출
                    if len(valid_indices) > 0:
                        segment_intensity_values = intensity.values[0, valid_indices]
                        max_idx_in_segment = np.argmax(segment_intensity_values)
                        target_time = times[valid_indices[max_idx_in_segment]]
                    else:
                        target_time = start_time + (segment_duration / 2)
                        
                    f1 = formant.get_value_at_time(1, target_time)
                    f2 = formant.get_value_at_time(2, target_time)
                    results.append({
                        "type": "monophthong",
                        "f1": float(f1) if not np.isnan(f1) else 0.0,
                        "f2": float(f2) if not np.isnan(f2) else 0.0,
                        "time": float(target_time)
                    })
        except Exception:
            pass # 단순 예외 처리
                
        return results

    def estimate_plosive_vot(self, audio_file: str, start_time: float = 0.0, end_time: float = None) -> float:
        """단순화된 파열음 VOT(Voice Onset Time) 자동 추정 로직 (ms 단위)"""
        import librosa
        import numpy as np
        
        y, sr = librosa.load(audio_file, sr=16000)
        
        if end_time is None:
            end_time = librosa.get_duration(y=y, sr=sr)
            
        start_frame = librosa.time_to_samples(start_time, sr=sr)
        end_frame = librosa.time_to_samples(end_time, sr=sr)
        y_seg = y[start_frame:end_frame]
        
        # Onset 탐지 (파열 버스트 시점 추정)
        onsets = librosa.onset.onset_detect(y=y_seg, sr=sr, units='time')
        
        if len(onsets) == 0:
            return 30.0 # 탐지 실패 시 기본 평음 수준의 VOT 반환
            
        burst_time = onsets[0]
        
        # 에너지(RMS) 변화를 통해 유성음(Voicing) 시작 시점 추정
        rms = librosa.feature.rms(y=y_seg)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        burst_frame = librosa.time_to_frames(burst_time, sr=sr)
        
        voicing_onset = burst_time
        # 버스트 시점 이후 에너지가 급격히 증가하는 구간을 성대 진동(Voicing) 시작으로 간주
        for i in range(burst_frame + 1, len(rms)):
            if rms[i] > rms[burst_frame] * 1.5: 
                voicing_onset = times[i]
                break
                
        vot_ms = (voicing_onset - burst_time) * 1000
        # 정상적인 VOT 범위 (0 ~ 150ms) 내로 클리핑
        return float(np.clip(vot_ms, 0.0, 150.0))

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
