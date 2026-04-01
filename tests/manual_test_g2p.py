import sys
from src.g2p_engine import KoryoG2PEngine

def test_convert_sentence_to_phonemes():
    try:
        g2p = KoryoG2PEngine()
        sentence = "국물 같이 먹자"
        result = g2p.convert(sentence)
        print(f"Input: {sentence}, Output: {result}")
        assert "궁물" in result
        assert "가치" in result
        print("Test 1 Passed!")
        
        sentence = "학교"
        result = g2p.convert(sentence)
        print(f"Input: {sentence}, Output: {result}")
        assert result == "학꾜"
        print("Test 2 Passed!")
        return True
    except Exception as e:
        print(f"Test Failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_convert_sentence_to_phonemes()
    sys.exit(0 if success else 1)
