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
from src.whisperx_processor import WhisperXProcessor

class PronunciationApp:
    """Integrated pronunciation correction application for Koryo-saram."""
    
    def __init__(self) -> None:
        self.g2p = KoryoG2PEngine()
        self.audio_proc = AudioProcessor()
        self.analyzer = AcousticAnalyzer()
        self.scorer = PronunciationScorer()
        self.whisperx_proc = None

    def analyze_with_whisperx(self, audio_path: str, target_text: str) -> Dict[str, Any]:
        """Analyzes pronunciation using WhisperX for transcription comparison."""
        if self.whisperx_proc is None:
            self.whisperx_proc = WhisperXProcessor()

        result = self.whisperx_proc.transcribe_and_align(audio_path)

        recognized_text = result.get("text", "")
        error = result.get("error", None)

        feedback_list = []
        raw_analysis = []
        total_score = 0.0

        if error:
            return {
                "target_text": target_text,
                "target_phonemes": self.g2p.convert(target_text),
                "total_score": 0.0,
                "feedback_details": [f"Analysis Error: {error}"],
                "analysis_raw": []
            }

        word_segments = result.get("word_segments", [])

        scores = []
        for w_seg in word_segments:
            word = w_seg.get("word", "")
            score = w_seg.get("score", 0.0)

            # Map score from 0.0-1.0 to 0-100
            word_score = score * 100.0
            scores.append(word_score)

            raw_analysis.append({
                "word": word,
                "confidence": score,
                "start": w_seg.get("start", 0.0),
                "end": w_seg.get("end", 0.0),
                "chars": w_seg.get("chars", [])
            })

            if word_score < 70.0:
                feedback_list.append(f"['{word}'] 발음의 정확도가 낮습니다 (신뢰도: {word_score:.1f}%). 더 또렷하게 발음해 보세요.")

        if not word_segments:
             feedback_list.append("음성에서 단어를 인식하지 못했습니다.")
             total_score = 0.0
        else:
             total_score = sum(scores) / len(scores)

        # Check text match
        if recognized_text.replace(" ", "") != target_text.replace(" ", "") and recognized_text:
            feedback_list.append(f"목표 문장과 인식된 문장이 다릅니다. (인식된 문장: '{recognized_text}')")
            total_score = total_score * 0.8 # Penalty for mismatch

        if not feedback_list and total_score > 0:
            feedback_list.append("매우 훌륭한 발음입니다!")

        report = {
            "target_text": target_text,
            "target_phonemes": self.g2p.convert(target_text),
            "total_score": float(total_score),
            "feedback_details": feedback_list,
            "analysis_raw": raw_analysis
        }
        return report

    def analyze_hybrid(self, audio_path: str, target_text: str) -> Dict[str, Any]:
        """Hybrid analysis: Uses WhisperX for segmentation and Acoustic for precision.
        Improved with partial matching for real-world interview data.
        """
        # 1. Run WhisperX first
        if self.whisperx_proc is None:
            self.whisperx_proc = WhisperXProcessor()
        
        whisper_res = self.whisperx_proc.transcribe_and_align(audio_path)
        recognized_text = whisper_res.get("text", "").replace(" ", "").replace(".", "").strip()
        target_text_clean = target_text.replace(" ", "").replace(".", "").strip()
        
        # 2. Advanced Text Match (Inclusion & Similarity)
        # Check if target is contained in recognized (common in interviews)
        is_included = target_text_clean in recognized_text or recognized_text in target_text_clean
        
        # Simple similarity: ratio of length or characters (placeholder for fuzzy match)
        common_len = min(len(recognized_text), len(target_text_clean))
        match_ratio = 1.0 if is_included else (common_len / max(len(recognized_text), len(target_text_clean), 1))
        
        # 3. Acoustic Analysis (Step 4 & 5)
        acoustic_report = self.analyze_pronunciation(audio_path, target_text)
        
        # 4. Hybrid Scoring Logic
        whisper_score = 0.0
        word_segments = whisper_res.get("word_segments", [])
        if word_segments:
            # Score based on recognized word confidence
            whisper_score = sum([w.get("score", 0.0) for w in word_segments]) / len(word_segments) * 100.0
        
        # Weighting: WhisperX(Recognition) 60% + Acoustic(Physical) 40%
        # If included, we don't apply the mismatch penalty
        total_score = (whisper_score * 0.6) + (acoustic_report['total_score'] * 0.4)
        
        if not is_included and match_ratio < 0.5:
            total_score = total_score * match_ratio # Apply penalty only if significantly different
            match_status = "불일치 (Mismatch)"
        elif is_included:
            match_status = "부분 일치/포함 (Partial Match)"
        else:
            match_status = "유사함 (Similar)"
            total_score = total_score * 0.9 # Slight penalty for slight difference
        
        hybrid_feedback = []
        if is_included:
            hybrid_feedback.append(f"문장이 정확히 인식되었습니다. (추가 발화 포함됨)")
        
        hybrid_feedback.extend(acoustic_report['feedback_details'])
        
        return {
            "target_text": target_text,
            "recognized_text": whisper_res.get("text", ""),
            "total_score": float(total_score),
            "feedback_details": hybrid_feedback,
            "match_status": match_status,
            "whisper_details": whisper_res
        }

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
        print("Usage: python app.py <audio_path> <target_text> [--hybrid]")
    else:
        app = PronunciationApp()
        if len(sys.argv) > 3 and sys.argv[3] == "--hybrid":
            result = app.analyze_hybrid(sys.argv[1], sys.argv[2])
        else:
            result = app.analyze_pronunciation(sys.argv[1], sys.argv[2])
        print(json.dumps(result, indent=4, ensure_ascii=False))
