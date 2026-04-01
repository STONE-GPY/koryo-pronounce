import os
import numpy as np
import librosa
import soundfile as sf
from src.audio_processor import AudioProcessor

def test_audio_normalization_and_vad():
    # 1. 테스트용 더미 오디오 생성 (1초 정적, 1초 소음, 1초 정적)
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration)
    # 1초 지점부터 0.5초간 440Hz 사인파 생성 (음성 가정)
    audio = np.zeros_like(t)
    audio[sr:int(1.5*sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[sr:int(1.5*sr)])
    
    test_file = "data/test_dummy.wav"
    os.makedirs("data", exist_ok=True)
    sf.write(test_file, audio, sr)
    
    processor = AudioProcessor(target_sample_rate=sr)
    
    # 2. 로드 및 정규화 테스트
    processed_audio = processor.load_and_normalize(test_file)
    assert np.max(np.abs(processed_audio)) <= 1.0
    
    # 3. VAD (Voice Activity Detection) 테스트
    # 무음 구간이 제거되고 유효 구간만 남아야 함
    voiced_audio = processor.apply_vad(processed_audio)
    assert len(voiced_audio) < len(processed_audio)
    assert len(voiced_audio) > 0
    
    os.remove(test_file)
