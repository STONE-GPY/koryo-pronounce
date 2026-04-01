import librosa
import numpy as np

class AudioProcessor:
    """오디오 로딩, 정규화, VAD(Voice Activity Detection) 수행"""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate

    def load_and_normalize(self, file_path: str) -> np.ndarray:
        """오디오 로딩 및 Peak Normalization 수행"""
        audio, _ = librosa.load(file_path, sr=self.sr)
        if len(audio) == 0:
            return audio
        
        # Peak Normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def apply_vad(self, audio: np.ndarray, top_db=30) -> np.ndarray:
        """무음 구간 제거 (librosa.effects.trim 사용)"""
        # trim은 시작과 끝의 무음을 제거함
        # 더 정교하게 조각조각 남기려면 split을 사용해야 함
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """간단한 Spectral Subtraction 또는 Low-pass Filter (확장용)"""
        # 현재는 placeholder
        return audio
