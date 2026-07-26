from piper import PiperVoice
voice = PiperVoice.load("/home/super/piper-voice")
audio = voice.synthesize("Hello, world!")
with open("test2.wav", "wb") as f:
    f.write(audio)
