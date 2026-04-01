import os
import json
import numpy as np
import soundfile as sf
from app import PronunciationApp

def test_full_pipeline_analysis():
    # 1. 테스트용 오디오 생성
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    test_audio = "data/full_test.wav"
    os.makedirs("data", exist_ok=True)
    sf.write(test_audio, audio, sr)
    
    app = PronunciationApp()
    # "학교" 문장에 대해 분석 수행 (실제로는 정밀 정렬이 필요하므로 여기서는 구조적 성공 위주)
    result = app.analyze_pronunciation(test_audio, "학교")
    
    assert "total_score" in result
    assert "feedback_details" in result
    assert isinstance(result["total_score"], (int, float))
    
    os.remove(test_audio)
