import os
import sys
from gtts import gTTS
from pydub import AudioSegment

def generate_sample_audio(text, output_filename):
    print(f"Generating audio for text: '{text}'")
    tts = gTTS(text=text, lang='ko')
    mp3_path = output_filename.replace('.wav', '.mp3')
    tts.save(mp3_path)
    
    # Convert mp3 to wav
    audio = AudioSegment.from_mp3(mp3_path)
    # Convert to 16kHz mono for our processor
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_filename, format="wav")
    
    os.remove(mp3_path)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_audio.py <text>")
        sys.exit(1)
    
    text = sys.argv[1]
    output_filename = "data/sample.wav"
    os.makedirs("data", exist_ok=True)
    generate_sample_audio(text, output_filename)
