
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from app import PronunciationApp

app = PronunciationApp()
target_text = "안녕하세요" # Assumed, as it's a common sample
audio_path = "data/sample.wav"

if os.path.exists(audio_path):
    report = app.analyze_pronunciation(audio_path, target_text)
    print("--- Analysis Report (Current Branch - Acoustic) ---")
    print(f"Target Text: {report['target_text']}")
    print(f"Total Score: {report['total_score']:.1f}")
    print(f"Feedback: {report['feedback_details']}")
    # print(f"Raw Analysis: {report['analysis_raw']}")
else:
    print(f"File not found: {audio_path}")
