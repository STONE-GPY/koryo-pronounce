import parselmouth
import numpy as np
import librosa
from typing import Dict, List, Any, Optional
from src.config import AudioConfig, PraatConfig, VotConfig, PitchConfig

class AcousticAnalyzer:
    """Handles acoustic analysis (Formant, VOT)."""
    
    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass

    def get_formants(self, audio_file: str) -> Dict[str, float]:
        """Extract F1 and F2 at the point of maximum intensity using Parselmouth (Praat)."""
        sound = parselmouth.Sound(audio_file)
        formant = sound.to_formant_burg(
            time_step=PraatConfig.TIME_STEP, 
            max_number_of_formants=PraatConfig.MAX_FORMANTS, 
            maximum_formant=PraatConfig.MAX_FORMANT_FREQ, 
            window_length=PraatConfig.WINDOW_LENGTH, 
            pre_emphasis_from=PraatConfig.PRE_EMPHASIS
        )
        
        try:
            intensity = sound.to_intensity()
            max_frame_idx = int(np.argmax(intensity.values[0]))
            target_time = intensity.get_time_from_frame_number(max_frame_idx + 1)
        except Exception as e:
            print(f"Intensity extraction failed: {e}")
            target_time = sound.get_total_duration() / 2.0
        
        f1 = formant.get_value_at_time(1, target_time)
        f2 = formant.get_value_at_time(2, target_time)
        
        return {
            "f1": float(f1) if not np.isnan(f1) else 0.0,
            "f2": float(f2) if not np.isnan(f2) else 0.0,
            "time": float(target_time)
        }

    def get_formants_for_segments(self, audio_file: str, vowel_types: List[str]) -> List[Dict[str, Any]]:
        """Extracts formants for multiple vowel segments."""
        num_segments = len(vowel_types)
        if num_segments <= 0:
            return []
            
        sound = parselmouth.Sound(audio_file)
        formant = sound.to_formant_burg(
            time_step=PraatConfig.TIME_STEP, 
            max_number_of_formants=PraatConfig.MAX_FORMANTS, 
            maximum_formant=PraatConfig.MAX_FORMANT_FREQ, 
            window_length=PraatConfig.WINDOW_LENGTH, 
            pre_emphasis_from=PraatConfig.PRE_EMPHASIS
        )
        duration = sound.get_total_duration()
        results: List[Dict[str, Any]] = []
        
        try:
            intensity = sound.to_intensity()
            times = intensity.xs()
            segment_duration = duration / num_segments
            
            for i, v_type in enumerate(vowel_types):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                valid_indices = np.where((times >= start_time) & (times < end_time))[0]
                
                if v_type == 'diphthong':
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
        except Exception as e:
            print(f"Error extracting formants for segments: {e}")
                
        return results

    def estimate_plosive_vot(self, audio_file: str, start_time: float = 0.0, end_time: Optional[float] = None) -> float:
        """Estimates the Voice Onset Time (VOT) for a plosive segment."""
        y, sr = librosa.load(audio_file, sr=AudioConfig.SAMPLE_RATE)
        
        if end_time is None:
            end_time = librosa.get_duration(y=y, sr=sr)
            
        start_frame = librosa.time_to_samples(start_time, sr=sr)
        end_frame = librosa.time_to_samples(end_time, sr=sr)
        y_seg = y[start_frame:end_frame]
        
        onsets = librosa.onset.onset_detect(y=y_seg, sr=sr, units='time')
        
        if len(onsets) == 0:
            return float(VotConfig.DEFAULT_VOT_ON_FAIL)
            
        burst_time = onsets[0]
        
        rms = librosa.feature.rms(y=y_seg)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        burst_frame = librosa.time_to_frames(burst_time, sr=sr)
        
        voicing_onset = burst_time
        for i in range(burst_frame + 1, len(rms)):
            if rms[i] > rms[burst_frame] * VotConfig.RMS_MULTIPLIER_THRESHOLD: 
                voicing_onset = times[i]
                break
                
        vot_ms = (voicing_onset - burst_time) * 1000.0
        return float(np.clip(vot_ms, 0.0, VotConfig.MAX_VOT_MS))

    def get_pitch(self, audio_file: str) -> float:
        """Extracts the average pitch (F0) from the audio."""
        sound = parselmouth.Sound(audio_file)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) > 0:
            return float(np.mean(pitch_values))
        return 0.0

    def estimate_gender(self, audio_file: str) -> str:
        """Estimates the speaker's gender based on average pitch."""
        mean_pitch = self.get_pitch(audio_file)
        if mean_pitch == 0.0:
            return "unknown"
        return "female" if mean_pitch > PitchConfig.GENDER_THRESHOLD else "male"

    def analyze_vowel_space(self, f1: float, f2: float) -> str:
        """Analyzes the vowel space (placeholder)."""
        return "Not Implemented"
