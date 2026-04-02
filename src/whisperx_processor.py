import os
import whisperx
import torch
from typing import Dict, Any

class WhisperXProcessor:
    """Processes audio using WhisperX for transcription and alignment."""

    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """Initializes the WhisperX processor.

        Args:
            model_name (str): Whisper model to load.
            device (str): Compute device ("cpu" or "cuda").
            compute_type (str): Compute precision type.
        """
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = device
            self.compute_type = compute_type

        print(f"Loading WhisperX model '{model_name}' on {self.device} with {self.compute_type}...")
        self.model = whisperx.load_model(model_name, self.device, compute_type=self.compute_type, language="ko")

    def transcribe_and_align(self, audio_path: str) -> Dict[str, Any]:
        """Transcribes and aligns the given audio file using WhisperX.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Dict[str, Any]: A dictionary containing the recognized text and aligned word segments.
        """
        if not os.path.exists(audio_path):
            return {"text": "", "segments": [], "error": f"Audio file not found: {audio_path}"}

        try:
            # 1. Transcribe with Whisper
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=4) # You can adjust batch size

            if not result["segments"]:
                 return {"text": "", "segments": []}

            recognized_text = " ".join([seg["text"] for seg in result["segments"]])

            # 2. Align whisper output (optional, but highly recommended for pronunciation tasks)
            # Alignment requires a separate model, often loaded dynamically based on language
            model_a, metadata = whisperx.load_align_model(language_code="ko", device=self.device)
            result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=True)

            return {
                "text": recognized_text,
                "segments": result_aligned["segments"],
                "word_segments": result_aligned["word_segments"]
            }

        except Exception as e:
            print(f"WhisperX transcription error: {e}")
            return {"text": "", "segments": [], "error": str(e)}
