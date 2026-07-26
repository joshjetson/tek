# Save as vosk_test.py
from vosk import Model, KaldiRecognizer
import pyaudio

model = Model("/home/super/vosk-model")
recognizer = KaldiRecognizer(model, 16000)
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

print("Say something...")
while True:
    data = stream.read(4096, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        print(recognizer.Result())
