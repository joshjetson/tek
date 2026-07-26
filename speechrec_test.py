# Save as speechrec_test.py
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone(device_index=11) as source:  # hw:2,0
    print("Say something...")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, timeout=5)
    try:
        print("You said:", r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Error: {e}")
