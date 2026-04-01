import os
import sys
import json
import uuid
import jamo
import librosa
import soundfile as sf
from typing import Dict, Any, List

from src.g2p_engine import KoryoG2PEngine
from src.audio_processor import AudioProcessor
from src.acoustic_analyzer import AcousticAnalyzer
from src.scorer import PronunciationScorer
from src.config import AudioConfig

class PronunciationApp:
    """Integrated pronunciation correction application for Koryo-saram."""
    
    def __init__(self) -> None:
        self.g2p = KoryoG2PEngine()
        self.audio_proc = AudioProcessor()
        self.analyzer = AcousticAnalyzer()
        self.scorer = PronunciationScorer()

    def analyze_pronunciation(self, audio_path: str, target_text: str) -> Dict[str, Any]:
        """Analyzes pronunciation by comparing the audio with the target text."""
        # 1. G2P Conversion
        phonemes = self.g2p.convert(target_text)
        phonemes_no_space = phonemes.replace(" ", "")
        
        # 2. Audio Preprocessing
        normalized_audio = self.audio_proc.load_and_normalize(audio_path)
        
        # Save processed audio temporarily for analysis (prevents race condition)
        temp_path = f"data/temp_processed_{uuid.uuid4().hex}.wav"
        os.makedirs("data", exist_ok=True)
        sf.write(temp_path, normalized_audio, AudioConfig.SAMPLE_RATE)
        
        # 3. Acoustic Analysis and Scoring (Syllable synchronization and diphthong support)
        # Measure user Pitch(F0) for personalized scaling
        user_pitch = self.analyzer.get_pitch(temp_path)
        
        duration = librosa.get_duration(path=temp_path)
        num_syllables = len(phonemes_no_space) # Number of syllables excluding spaces
        segment_duration = duration / num_syllables if num_syllables > 0 else duration
        
        scores: List[float] = []
        feedback_list: List[str] = []
        raw_analysis: List[Dict[str, Any]] = []
        
        vowel_types: List[str] = []
        syllable_vowels: List[str] = []
        syllable_plosives: List[str] = []
        
        for char in phonemes_no_space:
            decomposed = jamo.j2hcj(jamo.h2j(char))
            
            # Search for vowels (diphthongs prioritized)
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
            syllable_vowels.append(v_char if v_char else "")
            
            # Search for plosives
            p_char = None
            for p in decomposed:
                if p in self.scorer.vot_standards:
                    p_char = p
                    break
            syllable_plosives.append(p_char if p_char else "")
            
        segments_formants = self.analyzer.get_formants_for_segments(temp_path, vowel_types)
        
        for i in range(num_syllables):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration
            
            # Plosive scoring (independent matching per syllable)
            p_char = syllable_plosives[i]
            if p_char:
                vot_ms = self.analyzer.estimate_plosive_vot(temp_path, start_time, end_time)
                plosive_score = self.scorer.score_plosive(p_char, vot_ms)
                scores.append(float(plosive_score["score"]))
                feedback_list.append(str(plosive_score["feedback"]))
                raw_analysis.append({"phoneme": p_char, "vot_ms": vot_ms, "segment": i+1})
                
            # Vowel scoring (diphthong/monophthong distinction)
            v_char = syllable_vowels[i]
            v_type = vowel_types[i]
            if v_char and i < len(segments_formants):
                seg_data = segments_formants[i]
                if v_type == "diphthong":
                    score_res = self.scorer.score_diphthong(v_char, seg_data["start_f1"], seg_data["start_f2"], seg_data["end_f1"], seg_data["end_f2"], user_pitch)
                    scores.append(float(score_res["score"]))
                    feedback_list.append(str(score_res["feedback"]))
                    raw_analysis.append({"phoneme": v_char, "start_f1": seg_data["start_f1"], "end_f1": seg_data["end_f1"], "segment": i+1})
                else:
                    score_res = self.scorer.score_vowel(v_char, seg_data["f1"], seg_data["f2"], user_pitch)
                    scores.append(float(score_res["score"]))
                    feedback_list.append(str(score_res["feedback"]))
                    raw_analysis.append({"phoneme": v_char, "f1": seg_data["f1"], "f2": seg_data["f2"], "segment": i+1})
            
        if len(scores) > 0:
            total_score = sum(scores) / len(scores)
        else:
            total_score = 100.0
            feedback_list.append("No analyzable phonemes found.")
        
        # 5. Build final report
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
    # CLI execution example
    if len(sys.argv) < 3:
        print("Usage: python app.py <audio_path> <target_text>")
    else:
        app = PronunciationApp()
        result = app.analyze_pronunciation(sys.argv[1], sys.argv[2])
        print(json.dumps(result, indent=4, ensure_ascii=False))
