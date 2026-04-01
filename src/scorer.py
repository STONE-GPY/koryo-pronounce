class PronunciationScorer:
    """음성학적 수치를 기반으로 발음 점수 및 피드백 산출"""
    def __init__(self):
        # 파열음 표준 VOT (ms 단위)
        self.vot_standards = {
            "ㄲ": (0, 15, "경음"), # 매우 짧음
            "ㄱ": (35, 55, "평음"), # 중간
            "ㅋ": (80, 120, "격음") # 매우 김
        }
        # 모음 표준 포먼트 (F1, F2 - 성인 남성 평균 예시)
        self.vowel_standards = {
            "ㅏ": (750, 1250),
            "ㅣ": (300, 2200),
            "ㅜ": (350, 800)
        }

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

    def score_vowel(self, target_phoneme: str, user_f1: float, user_f2: float) -> dict:
        """모음의 포먼트(F1, F2) 위치 점수화"""
        if target_phoneme not in self.vowel_standards:
            return {"score": 100, "feedback": "준비되지 않은 음소입니다."}
            
        std_f1, std_f2 = self.vowel_standards[target_phoneme]
        
        # 유클리드 거리 기반 점수 산출
        dist = ((user_f1 - std_f1)**2 + (user_f2 - std_f2)**2)**0.5
        score = max(0, 100 - (dist / 10)) # 거리 1000이면 0점
        
        return {
            "score": score, 
            "feedback": f"'{target_phoneme}' 모음의 표준 포먼트 대비 거리는 {dist:.1f}입니다."
        }
