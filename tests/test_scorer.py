import pytest
from src.scorer import PronunciationScorer

def test_calculate_score_for_plosives():
    scorer = PronunciationScorer()
    
    # Assuming 'ㄱ' (lax) standard VOT is around 35~55ms
    # 10ms (e.g. Russian-like pronunciation) should get a lower score
    score_info = scorer.score_plosive(target_phoneme="ㄱ", user_vot=10)
    assert score_info["score"] < 70
    assert "Release force is weak" in score_info["feedback"]
    
    # Normal range score
    score_info_normal = scorer.score_plosive(target_phoneme="ㄱ", user_vot=40)
    assert score_info_normal["score"] >= 90

def test_vowel_formant_scoring():
    scorer = PronunciationScorer()
    # For 'ㅏ' (standard F1~750, F2~1250)
    # 750, 1150 should get a decent score
    score_info = scorer.score_vowel(target_phoneme="ㅏ", user_f1=750, user_f2=1150)
    assert score_info["score"] > 80
    
    # Extremely off formants (e.g., trying to say 'ㅏ' but it sounds like 'ㅜ' with 300, 800)
    score_info_wrong = scorer.score_vowel(target_phoneme="ㅏ", user_f1=300, user_f2=800)
    assert score_info_wrong["score"] < 60
