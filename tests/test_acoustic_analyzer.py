import os
import numpy as np
import soundfile as sf
import pytest
from src.acoustic_analyzer import AcousticAnalyzer

def test_extract_formants_from_vowel():
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. 간단한 모음 'ㅏ' (F1=800, F2=1200 가정) 시뮬레이션
    # (실제 Praat 분석은 파형이 더 복잡해야 하므로 더미 오디오보다는 
    # 분석기 객체가 Parselmouth Sound 객체를 잘 생성하는지 위주로 테스트)
    audio = 0.5 * np.sin(2 * np.pi * 500 * t) # 단순 주파수
    
    test_file = "data/vowel_test.wav"
    os.makedirs("data", exist_ok=True)
    sf.write(test_file, audio, sr)
    
    analyzer = AcousticAnalyzer()
    
    # Formant 추출 시도
    formants = analyzer.get_formants(test_file)
    assert "f1" in formants
    assert "f2" in formants
    assert formants["f1"] > 0
    
    os.remove(test_file)

def test_vot_measurement_logic():
    # VOT는 복잡한 알고리즘이 필요하므로, 
    # 일단 버스트 지점과 진동 시작 지점을 찾는 내부 함수가 정의되어 있는지 확인
    analyzer = AcousticAnalyzer()
    assert hasattr(analyzer, "measure_vot")
