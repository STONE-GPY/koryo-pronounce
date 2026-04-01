from src.config import PitchConfig, ScoringConfig

class PronunciationScorer:
    """음성학적 수치를 기반으로 발음 점수 및 피드백 산출"""
    def __init__(self):
        # 파열음 표준 VOT (ms 단위)
        self.vot_standards = {
            "ㄲ": (0, 15, "경음"), # 매우 짧음
            "ㄱ": (35, 55, "평음"), # 중간
            "ㅋ": (80, 120, "격음") # 매우 김
        }
        # 모음 표준 포먼트 (기본값) - 개인화 스케일링의 기준점이 됨
        self.base_vowel_standards = {
            "ㅏ": (750, 1250),
            "ㅓ": (600, 1000),
            "ㅗ": (400, 850),
            "ㅜ": (350, 800),
            "ㅡ": (350, 1300),
            "ㅣ": (300, 2200),
            "ㅔ": (550, 1700),
            "ㅐ": (600, 1600)
        }
        # 이중모음 (시작점(F1, F2), 종료점(F1, F2))
        self.base_diphthong_standards = {
            "ㅑ": ((300, 2200), (750, 1250)),
            "ㅕ": ((300, 2200), (600, 1000)),
            "ㅛ": ((300, 2200), (400, 850)),
            "ㅠ": ((300, 2200), (350, 800)),
            "ㅘ": ((400, 850), (750, 1250)),
            "ㅝ": ((350, 800), (600, 1000)),
            "ㅙ": ((400, 850), (600, 1600)),
            "ㅞ": ((350, 800), (550, 1700)),
            "ㅚ": ((400, 850), (550, 1700)),
            "ㅟ": ((350, 800), (300, 2200)),
            "ㅢ": ((350, 1300), (300, 2200)),
            "ㅒ": ((300, 2200), (600, 1600)),
            "ㅖ": ((300, 2200), (550, 1700))
        }

    @property
    def vowel_standards(self):
        """음소 존재 여부 검사를 위한 기본 키 집합 반환"""
        return list(self.base_vowel_standards.keys())
        
    @property
    def diphthong_standards(self):
        return list(self.base_diphthong_standards.keys())

    def score_plosive(self, target_phoneme: str, user_vot: float) -> dict:
        """파열음(ㄱ, ㄲ, ㅋ 등)의 VOT 점수화"""
        if target_phoneme not in self.vot_standards:
            return {"score": 100, "feedback": "준비되지 않은 음소입니다."}
            
        std_min, std_max, p_type = self.vot_standards[target_phoneme]
        mid = (std_min + std_max) / 2
        diff = abs(user_vot - mid)
        
        # 0 ~ 100 점수 산출 (가우시안 혹은 선형 감점)
        # 차이가 40ms 이상이면 0점 근처
        score = max(0, 100 - (diff * 2))
        
        feedback = f"{p_type} 발음의 VOT는 표준 {std_min}~{std_max}ms이나, 사용자는 {user_vot:.1f}ms입니다."
        
        if user_vot < std_min:
            feedback += " 공기가 터지는 시간이 너무 빠릅니다."
        elif user_vot > std_max:
            feedback += " 공기가 터지는 시간이 너무 늦습니다."
        else:
            feedback += " 아주 정확한 타이밍입니다!"
            
        return {"score": score, "feedback": feedback}

    def score_vowel(self, target_phoneme: str, user_f1: float, user_f2: float, user_pitch: float = 0.0) -> dict:
        """Pitch(F0) 기반 동적 포먼트 스케일링 적용 (개인 맞춤형 모음 점수화)"""
        if target_phoneme not in self.base_vowel_standards:
            return {"score": 100, "feedback": "준비되지 않은 음소입니다."}
            
        base_f1, base_f2 = self.base_vowel_standards[target_phoneme]
        
        # 개인화 스케일링 (Vocal Tract Length Normalization Approximation)
        if user_pitch > PitchConfig.MIN_VALID_PITCH:
            scale_factor = 1.0 + ((user_pitch - PitchConfig.MALE_BASE_PITCH) * ScoringConfig.SCALE_FACTOR_SLOPE)
            scale_factor = max(ScoringConfig.MIN_SCALE_FACTOR, min(ScoringConfig.MAX_SCALE_FACTOR, scale_factor))
        else:
            scale_factor = 1.0
            
        target_f1 = base_f1 * scale_factor
        target_f2 = base_f2 * scale_factor
        
        # 유클리드 거리 기반 점수 산출
        dist = ((user_f1 - target_f1)**2 + (user_f2 - target_f2)**2)**0.5
        score = max(0, 100 - (dist / ScoringConfig.VOWEL_PENALTY_DIVISOR)) 
        
        feedback = (f"[개인화 타겟 F1:{target_f1:.0f} F2:{target_f2:.0f} (Pitch:{user_pitch:.0f}Hz)] "
                    f"'{target_phoneme}' 모음의 기준 대비 오차 거리는 {dist:.1f}입니다.")
        
        return {"score": score, "feedback": feedback}

    def score_diphthong(self, target_phoneme: str, user_start_f1: float, user_start_f2: float, user_end_f1: float, user_end_f2: float, user_pitch: float = 0.0) -> dict:
        """이중모음의 시작점과 종료점 포먼트를 기반으로 점수 산출"""
        if target_phoneme not in self.base_diphthong_standards:
            return {"score": 100, "feedback": "준비되지 않은 음소입니다."}
            
        base_start, base_end = self.base_diphthong_standards[target_phoneme]
        
        if user_pitch > PitchConfig.MIN_VALID_PITCH:
            scale_factor = 1.0 + ((user_pitch - PitchConfig.MALE_BASE_PITCH) * ScoringConfig.SCALE_FACTOR_SLOPE)
            scale_factor = max(ScoringConfig.MIN_SCALE_FACTOR, min(ScoringConfig.MAX_SCALE_FACTOR, scale_factor))
        else:
            scale_factor = 1.0
            
        target_start = (base_start[0] * scale_factor, base_start[1] * scale_factor)
        target_end = (base_end[0] * scale_factor, base_end[1] * scale_factor)
        
        dist_start = ((user_start_f1 - target_start[0])**2 + (user_start_f2 - target_start[1])**2)**0.5
        dist_end = ((user_end_f1 - target_end[0])**2 + (user_end_f2 - target_end[1])**2)**0.5
        avg_dist = (dist_start + dist_end) / 2
        
        score = max(0, 100 - (avg_dist / ScoringConfig.VOWEL_PENALTY_DIVISOR))
        
        feedback = (f"[개인화 이중모음 타겟 (Pitch:{user_pitch:.0f}Hz)] "
                    f"'{target_phoneme}'의 시작점 오차: {dist_start:.1f}, 종료점 오차: {dist_end:.1f} (평균 오차: {avg_dist:.1f})")
        return {"score": score, "feedback": feedback}