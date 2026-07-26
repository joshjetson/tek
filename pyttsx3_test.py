# pyttsx3_test_voices.py
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Testing voice: {voice.id}")
    engine.setProperty('voice', voice.id)
    engine.say("Testing voice")
    engine.runAndWait()
