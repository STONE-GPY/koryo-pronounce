from typing import Dict, List, Tuple, Union, Optional
from src.config import PitchConfig, ScoringConfig

class PronunciationScorer:
    """Calculates pronunciation scores and provides feedback based on acoustic features."""
    
    def __init__(self) -> None:
        """Initialize the scorer with standard phonetic values."""
        # Plosive standard VOT (in ms)
        self.vot_standards: Dict[str, Tuple[float, float, str]] = {
            "ㄲ": (0.0, 15.0, "경음"),  # Tense (very short)
            "ㄱ": (35.0, 55.0, "평음"),  # Lax (medium)
            "ㅋ": (80.0, 120.0, "격음")  # Aspirated (very long)
        }
        
        # Base vowel standards (F1, F2) - used as baseline for personalized scaling
        self.base_vowel_standards: Dict[str, Tuple[float, float]] = {
            "ㅏ": (750.0, 1250.0),
            "ㅓ": (600.0, 1000.0),
            "ㅗ": (400.0, 850.0),
            "ㅜ": (350.0, 800.0),
            "ㅡ": (350.0, 1300.0),
            "ㅣ": (300.0, 2200.0),
            "ㅔ": (550.0, 1700.0),
            "ㅐ": (600.0, 1600.0)
        }
        
        # Diphthong standards ((start_F1, start_F2), (end_F1, end_F2))
        self.base_diphthong_standards: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
            "ㅑ": ((300.0, 2200.0), (750.0, 1250.0)),
            "ㅕ": ((300.0, 2200.0), (600.0, 1000.0)),
            "ㅛ": ((300.0, 2200.0), (400.0, 850.0)),
            "ㅠ": ((300.0, 2200.0), (350.0, 800.0)),
            "ㅘ": ((400.0, 850.0), (750.0, 1250.0)),
            "ㅝ": ((350.0, 800.0), (600.0, 1000.0)),
            "ㅙ": ((400.0, 850.0), (600.0, 1600.0)),
            "ㅞ": ((350.0, 800.0), (550.0, 1700.0)),
            "ㅚ": ((400.0, 850.0), (550.0, 1700.0)),
            "ㅟ": ((350.0, 800.0), (300.0, 2200.0)),
            "ㅢ": ((350.0, 1300.0), (300.0, 2200.0)),
            "ㅒ": ((300.0, 2200.0), (600.0, 1600.0)),
            "ㅖ": ((300.0, 2200.0), (550.0, 1700.0))
        }

    @property
    def vowel_standards(self) -> List[str]:
        """Returns the list of supported monophthongs."""
        return list(self.base_vowel_standards.keys())
        
    @property
    def diphthong_standards(self) -> List[str]:
        """Returns the list of supported diphthongs."""
        return list(self.base_diphthong_standards.keys())

    def _get_scale_factor(self, user_pitch: float) -> float:
        """Calculates scaling factor for personalization based on user pitch."""
        if user_pitch > PitchConfig.MIN_VALID_PITCH:
            scale_factor = 1.0 + ((user_pitch - PitchConfig.MALE_BASE_PITCH) * ScoringConfig.SCALE_FACTOR_SLOPE)
            return max(ScoringConfig.MIN_SCALE_FACTOR, min(ScoringConfig.MAX_SCALE_FACTOR, scale_factor))
        return 1.0

    def score_plosive(self, target_phoneme: str, user_vot: float) -> Dict[str, Union[float, str]]:
        """Scores Voice Onset Time (VOT) for plosives (e.g., ㄱ, ㄲ, ㅋ)."""
        if target_phoneme not in self.vot_standards:
            return {"score": 0.0, "feedback": f"Unsupported phoneme: {target_phoneme}"}
            
        std_min, std_max, p_type = self.vot_standards[target_phoneme]
        mid = (std_min + std_max) / 2
        diff = abs(user_vot - mid)
        
        # 0 ~ 100 score calculation (linear penalty)
        score = max(0.0, 100.0 - (diff * 2.0))
        
        feedback = f"[{target_phoneme} pronunciation] "
        
        if user_vot < std_min:
            feedback += "Release force is weak. Apply more pressure to the articulators and release with more air."
        elif user_vot > std_max:
            if p_type == "경음":
                feedback += "Too much air leakage. Tense the throat and release the sound more abruptly."
            else:
                feedback += "Aspiration is too long. Try to make the burst shorter."
        else:
            feedback += "Excellent timing!"
            
        feedback += f" (Accuracy: {score:.1f}%)"
        return {"score": float(score), "feedback": feedback}

    def score_vowel(self, target_phoneme: str, user_f1: float, user_f2: float, user_pitch: float = 0.0) -> Dict[str, Union[float, str]]:
        """Scores monophthongs using F1/F2 formants with dynamic scaling."""
        if target_phoneme not in self.base_vowel_standards:
            return {"score": 0.0, "feedback": f"Unsupported vowel: {target_phoneme}"}
            
        base_f1, base_f2 = self.base_vowel_standards[target_phoneme]
        scale_factor = self._get_scale_factor(user_pitch)
            
        target_f1 = base_f1 * scale_factor
        target_f2 = base_f2 * scale_factor
        
        # Euclidean distance based scoring
        dist = ((user_f1 - target_f1)**2 + (user_f2 - target_f2)**2)**0.5
        score = max(0.0, 100.0 - (dist / ScoringConfig.VOWEL_PENALTY_DIVISOR)) 
        
        f1_diff = target_f1 - user_f1
        f2_diff = target_f2 - user_f2
        
        anatomical_feedback = ""
        if score < 90:
            if f1_diff > 50:
                anatomical_feedback += "Open your mouth wider. "
            elif f1_diff < -50:
                anatomical_feedback += "Close your mouth a bit more. "
                
            if f2_diff > 100:
                anatomical_feedback += "Move your tongue forward. "
            elif f2_diff < -100:
                anatomical_feedback += "Move your tongue back. "
                
        if anatomical_feedback:
            feedback = f"[{target_phoneme} correction] {anatomical_feedback.strip()} (Accuracy: {score:.1f}%)"
        else:
            feedback = f"[{target_phoneme} pronunciation] Great vowel pronunciation! (Accuracy: {score:.1f}%)"
        
        return {"score": float(score), "feedback": feedback}

    def score_diphthong(self, target_phoneme: str, user_start_f1: float, user_start_f2: float, 
                        user_end_f1: float, user_end_f2: float, user_pitch: float = 0.0) -> Dict[str, Union[float, str]]:
        """Scores diphthongs based on start and end formant transitions."""
        if target_phoneme not in self.base_diphthong_standards:
            return {"score": 0.0, "feedback": f"Unsupported diphthong: {target_phoneme}"}
            
        base_start, base_end = self.base_diphthong_standards[target_phoneme]
        scale_factor = self._get_scale_factor(user_pitch)
            
        target_start = (base_start[0] * scale_factor, base_start[1] * scale_factor)
        target_end = (base_end[0] * scale_factor, base_end[1] * scale_factor)
        
        dist_start = ((user_start_f1 - target_start[0])**2 + (user_start_f2 - target_start[1])**2)**0.5
        dist_end = ((user_end_f1 - target_end[0])**2 + (user_end_f2 - target_end[1])**2)**0.5
        avg_dist = (dist_start + dist_end) / 2
        
        score = max(0.0, 100.0 - (avg_dist / ScoringConfig.VOWEL_PENALTY_DIVISOR))
        
        anatomical_feedback = ""
        if score < 90:
            if dist_start > dist_end:
                anatomical_feedback = "The starting position of the sound is inaccurate. Try to make the initial sound clearer."
            else:
                anatomical_feedback = "The ending position of the sound is inaccurate. Ensure smooth movement of tongue and lips until the end."
        
        if anatomical_feedback:
            feedback = f"[{target_phoneme} correction] {anatomical_feedback} (Accuracy: {score:.1f}%)"
        else:
            feedback = f"[{target_phoneme} pronunciation] Natural diphthong! (Accuracy: {score:.1f}%)"
            
        return {"score": float(score), "feedback": feedback}
