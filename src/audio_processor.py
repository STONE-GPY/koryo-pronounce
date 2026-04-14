import librosa
import numpy as np
import os
from scipy.signal import butter, lfilter
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
        if len(audio) == 0:
            return audio
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Performs noise reduction using a Band-pass filter and simple Spectral Subtraction.
        1. Band-pass filter (80Hz - 8000Hz) to focus on human voice.
        2. Spectral subtraction to reduce stationary background noise.
        """
        if len(audio) == 0:
            return audio

        # 1. Band-pass Filter (Focus on human voice range)
        nyquist = 0.5 * self.sr
        low = 80 / nyquist
        high = min(8000 / nyquist, 0.99) # Ensure it doesn't exceed Nyquist
        b, a = butter(1, [low, high], btype='band')
        filtered_audio = lfilter(b, a, audio)

        # 2. Simple Spectral Subtraction
        # Short-Time Fourier Transform (STFT)
        stft = librosa.stft(filtered_audio)
        magnitude, phase = librosa.magphase(stft)

        # Estimate noise floor (using the median magnitude across time as a heuristic)
        noise_est = np.median(magnitude, axis=1, keepdims=True)

        # Subtract noise with a safety floor (to prevent "musical noise" artifacts)
        # We use a multiplier (1.5) to be a bit more aggressive with background noise
        magnitude_clean = np.maximum(magnitude - 1.5 * noise_est, magnitude * 0.02)

        # Inverse STFT to get back to time domain
        audio_clean = librosa.istft(magnitude_clean * phase)
        
        # Ensure the length matches (istft might slightly change length)
        if len(audio_clean) > len(audio):
            audio_clean = audio_clean[:len(audio)]
        elif len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

        return audio_clean
