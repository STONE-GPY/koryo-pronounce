from src.g2p_engine import KoryoG2PEngine
import pytest

def test_convert_sentence_to_phonemes():
    g2p = KoryoG2PEngine()
    sentence = "국물 같이 먹자"
    # '국물' -> [궁물], '같이' -> [가치]
    result = g2p.convert(sentence)
    assert "궁물" in result
    assert "가치" in result

def test_romanization_for_russian_speakers():
    g2p = KoryoG2PEngine()
    sentence = "학교"
    # g2pk는 보통 표준 발음법을 따름
    result = g2p.convert(sentence)
    assert result == "학꾜"

def test_empty_string():
    g2p = KoryoG2PEngine()
    assert g2p.convert("") == ""

def test_get_phonemes():
    g2p = KoryoG2PEngine()
    sentence = "학교 가자"
    result = g2p.get_phoneme_list(sentence)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "학꾜"

def test_invalid_input():
    g2p = KoryoG2PEngine()
    assert g2p.convert(None) == ""
    assert g2p.convert(123) == ""
