import pytest
from src.scorer import PronunciationScorer

def test_calculate_score_for_plosives():
    scorer = PronunciationScorer()
    
    # 한국어 'ㄱ' (평음)의 표준 VOT가 30~50ms라고 가정할 때, 
    # 10ms (러시아어식 발음)이면 낮은 점수가 나와야 함
    score_info = scorer.score_plosive(target_phoneme="ㄱ", user_vot=10)
    assert score_info["score"] < 70
    assert "평음" in score_info["feedback"]
    
    # 정상 범위 점수
    score_info_normal = scorer.score_plosive(target_phoneme="ㄱ", user_vot=40)
    assert score_info_normal["score"] >= 90

def test_vowel_formant_scoring():
    scorer = PronunciationScorer()
    # 'ㅏ' 발음의 표준 F1=800, F2=1200일 때, 
    # 750, 1150이면 적절한 점수가 나와야 함
    score_info = scorer.score_vowel(target_phoneme="ㅏ", user_f1=750, user_f2=1150)
    assert score_info["score"] > 80
    
    # 너무 동떨어진 포먼트 (예: 'ㅏ'인데 'ㅜ'와 유사한 300, 800)
    score_info_wrong = scorer.score_vowel(target_phoneme="ㅏ", user_f1=300, user_f2=800)
    assert score_info_wrong["score"] < 60
