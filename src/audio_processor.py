import librosa
import numpy as np
import os
from src.config import AudioConfig

class AudioProcessor:
    """Performs audio loading, normalization, and Voice Activity Detection (VAD)."""
    def __init__(self, target_sample_rate: int = AudioConfig.SAMPLE_RATE):
        self.sr = target_sample_rate

    def load_and_normalize(self, file_path: str) -> np.ndarray:
        """Loads audio and performs Peak Normalization."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return np.array([], dtype=np.float32)
        
        try:
            audio, _ = librosa.load(file_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.array([], dtype=np.float32)

        if len(audio) == 0:
            return audio
        
        # Peak Normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def apply_vad(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Removes silent segments (using librosa.effects.trim)."""
        # trim removes silence from the beginning and end
        # To leave more sophisticated pieces, split should be used
        if len(audio) == 0:
            return audio
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """Basic Spectral Subtraction or Low-pass Filter (for extension)."""
        # Currently a placeholder
        return audio
