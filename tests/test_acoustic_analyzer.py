import os
import numpy as np
import soundfile as sf
import pytest
from src.acoustic_analyzer import AcousticAnalyzer

def test_extract_formants_from_vowel():
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Simple simulation for vowel (F1 ~800, F2 ~1200)
    audio = 0.5 * np.sin(2 * np.pi * 500 * t) 
    
    test_file = "data/vowel_test.wav"
    os.makedirs("data", exist_ok=True)
    sf.write(test_file, audio, sr)
    
    analyzer = AcousticAnalyzer()
    
    # Extract formants
    formants = analyzer.get_formants(test_file)
    assert "f1" in formants
    assert "f2" in formants
    assert formants["f1"] > 0.0
    
    # Test pitch extraction
    pitch = analyzer.get_pitch(test_file)
    assert pitch > 0.0
    
    # Test gender estimation
    gender = analyzer.estimate_gender(test_file)
    assert gender in ["male", "female", "unknown"]
    
    os.remove(test_file)

def test_vot_measurement_logic():
    analyzer = AcousticAnalyzer()
    assert hasattr(analyzer, "estimate_plosive_vot")
