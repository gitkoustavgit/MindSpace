import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import scipy.io.wavfile as wav

# Load Whisper model
model = whisper.load_model("base")

def get_voice_input(duration=5, samplerate=16000) -> str:
    """Record voice from mic and return transcribed text."""
    print("\n🎙️ Recording... Speak now")

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished")

    # Save to temp wav file
    tmpfile = tempfile.mktemp(suffix=".wav")
    wav.write(tmpfile, samplerate, audio)

    # Transcribe with Whisper
    result = model.transcribe(tmpfile)

    # Cleanup
    os.remove(tmpfile)

    text = result["text"].strip()
    print("Transcribed text:", text)
    return text