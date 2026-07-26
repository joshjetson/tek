#!/usr/bin/env python3
"""
Enhanced Voice Assistant with Multiple TTS Options
Jetson Nano 2GB - Better voice quality options
"""

import speech_recognition as sr
import subprocess
import sys
import os
import tempfile
import json
import time
import urllib.request
import urllib.parse

class EnhancedVoiceAssistant:
    def __init__(self):
        print("Initializing Enhanced Voice Assistant...")
        
        # Audio device configuration
        self.audio_device = "plughw:2,0"
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=11)
        
        # TTS options (in order of preference - Google TTS first!)
        self.tts_methods = {
            'google_tts': self.speak_google_tts,
            'festival': self.speak_festival,
            'espeak_enhanced': self.speak_espeak_enhanced,
            'espeak_default': self.speak_espeak_default
        }
        
        # Detect available TTS methods
        self.available_tts = self.detect_tts_methods()
        self.current_tts = self.available_tts[0] if self.available_tts else 'espeak_default'
        
        # Setup TinyLlama
        self.llama_executable = '/bin/ollama'
        
        print(f"Available TTS methods: {self.available_tts}")
        print(f"Using TTS method: {self.current_tts}")
        print("Enhanced Voice Assistant initialized!")
        
    def detect_tts_methods(self):
        """Detect which TTS methods are available"""
        available = []
        
        # Google TTS (requires internet)
        available.append('google_tts')
        print("✓ Google TTS available (requires internet)")
        
        # Check for Festival
        try:
            subprocess.run(['festival', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            available.append('festival')
            print("✓ Festival TTS detected")
        except:
            print("✗ Festival TTS not available")
        
        # Default espeak (always available)
        available.append('espeak_enhanced')  # Try enhanced settings first
        available.append('espeak_default')
        
        return available
    
    def speak_festival(self, text):
        """Use Festival TTS for more natural speech"""
        try:
            clean_text = text.replace('"', '').replace("'", "").strip()
            
            # Use temporary file method (this works!)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Generate WAV file with text2wave
            cmd = f'echo "{clean_text}" | text2wave -o {temp_filename}'
            result = subprocess.run(cmd, shell=True, timeout=10,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                # Play the WAV file through USB audio
                cmd = f'aplay -D {self.audio_device} {temp_filename}'
                subprocess.run(cmd, shell=True, check=True, timeout=10,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Clean up
                os.unlink(temp_filename)
                return True
            else:
                # Clean up on error
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                return False
                
        except Exception as e:
            print(f"Festival TTS error: {e}")
            return False
    
    def speak_espeak_enhanced(self, text):
        """Enhanced espeak with better voice settings"""
        try:
            clean_text = text.replace('"', '').replace("'", "").strip()
            
            # Try different voice options for more natural speech
            voice_options = [
                '-v en-us+f3 -s 160 -p 50 -a 180',  # Female voice, slower, varied pitch
                '-v en-gb+f4 -s 150 -p 45 -a 180',  # British female
                '-v en+m3 -s 155 -p 40 -a 180',     # Male voice variant
                '-v en-us -s 150 -p 50 -a 180'      # Default US with better settings
            ]
            
            for voice_setting in voice_options:
                try:
                    cmd = f'espeak {voice_setting} "{clean_text}" --stdout | aplay -D {self.audio_device}'
                    subprocess.run(cmd, shell=True, check=True, timeout=10)
                    return True
                except:
                    continue
                    
            # Fallback to default
            return self.speak_espeak_default(text)
            
        except Exception as e:
            print(f"Enhanced espeak error: {e}")
            return False
    
    def speak_google_tts(self, text):
        """Use Google Text-to-Speech (requires internet)"""
        try:
            clean_text = text.replace('"', '').replace("'", "").strip()
            
            # Create Google TTS URL
            base_url = "https://translate.google.com/translate_tts"
            params = {
                'ie': 'UTF-8',
                'q': clean_text,
                'tl': 'en',
                'client': 'tw-ob'
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            # Download audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                try:
                    # Set user agent to avoid blocking
                    req = urllib.request.Request(url, headers={
                        'User-Agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36'
                    })
                    
                    with urllib.request.urlopen(req, timeout=10) as response:
                        temp_file.write(response.read())
                    
                    temp_filename = temp_file.name
                
                    # Play the MP3 file through USB audio
                    # Try mpg123 first, then mpv, then ffmpeg
                    players = [
                        f'mpg123 -a {self.audio_device} "{temp_filename}"',
                        f'mpv --audio-device=alsa/{self.audio_device} --no-video "{temp_filename}"',
                        f'ffplay -nodisp -autoexit -i "{temp_filename}"'
                    ]
                    
                    for player_cmd in players:
                        try:
                            subprocess.run(player_cmd, shell=True, check=True, timeout=15,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            os.unlink(temp_filename)
                            return True
                        except:
                            continue
                    
                    # Cleanup if all players failed
                    os.unlink(temp_filename)
                    return False
                    
                except Exception as e:
                    print(f"Google TTS download error: {e}")
                    return False
                    
        except Exception as e:
            print(f"Google TTS error: {e}")
            return False
    
    def speak_espeak_default(self, text):
        """Fallback espeak with original settings"""
        try:
            clean_text = text.replace('"', '').replace("'", "").strip()
            cmd = f'espeak -a 200 -s 150 -v en-us "{clean_text}" --stdout | aplay -D {self.audio_device}'
            subprocess.run(cmd, shell=True, check=True, timeout=10,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            print(f"Default espeak error: {e}")
            return False
    
    def speak(self, text):
        """Main TTS function - tries methods in order of preference"""
        if not text or not text.strip():
            return
            
        print(f"Speaking: {text}")
        
        # Try current TTS method
        if self.current_tts in self.tts_methods:
            if self.tts_methods[self.current_tts](text):
                return
        
        # Try fallback methods
        for method in self.available_tts:
            if method != self.current_tts and method in self.tts_methods:
                if self.tts_methods[method](text):
                    return
        
        # Final fallback - just print
        print(f"TTS failed, text output: {text}")
    
    def change_voice(self, voice_type):
        """Change TTS method"""
        if voice_type in self.available_tts:
            self.current_tts = voice_type
            print(f"Changed voice to: {voice_type}")
            self.speak(f"Voice changed to {voice_type.replace('_', ' ')}")
        else:
            print(f"Voice type {voice_type} not available")
            print(f"Available voices: {self.available_tts}")
    
    def listen(self):
        """Listen for voice input"""
        try:
            print("Listening...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate AI response using Ollama"""
        try:
            prompt = f"You are a helpful assistant. Answer briefly and clearly.\n\nHuman: {user_input}\nAssistant:"
            
            cmd = ['/bin/ollama', 'run', 'tinyllama', prompt]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=30
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    if "Assistant:" in response:
                        response = response.split("Assistant:")[-1].strip()
                    return response[:200]
                else:
                    return "I'm not sure how to respond to that."
            else:
                return "I'm having trouble processing that request."
                
        except subprocess.TimeoutExpired:
            return "I'm thinking too hard. Could you try again?"
        except Exception as e:
            print(f"AI generation error: {e}")
            return "I'm having trouble processing that request."
    
    def handle_voice_commands(self, user_input):
        """Handle voice/TTS related commands"""
        user_input = user_input.lower().strip()
        
        if 'change voice' in user_input or 'switch voice' in user_input:
            if 'festival' in user_input:
                self.change_voice('festival')
            elif 'google' in user_input:
                self.change_voice('google_tts')
            elif 'enhanced' in user_input:
                self.change_voice('espeak_enhanced')
            elif 'default' in user_input:
                self.change_voice('espeak_default')
            else:
                available_voices = ', '.join(self.available_tts)
                self.speak(f"Available voices are: {available_voices}. Say change voice followed by the voice name.")
            return True
        
        if 'list voices' in user_input or 'available voices' in user_input:
            available_voices = ', '.join(self.available_tts)
            self.speak(f"Available voices are: {available_voices}")
            return True
            
        if 'current voice' in user_input:
            self.speak(f"Current voice is {self.current_tts.replace('_', ' ')}")
            return True
            
        return False
    
    def handle_simple_commands(self, user_input):
        """Handle simple commands without AI"""
        user_input = user_input.lower().strip()
        
        # Voice commands first
        if self.handle_voice_commands(user_input):
            return True
        
        # Time commands
        if any(word in user_input for word in ['time', 'clock', 'what time']):
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"It's {current_time}"
        
        # Date commands
        if any(word in user_input for word in ['date', 'today', 'what day']):
            import datetime
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {today}"
        
        # Math (simple)
        if user_input in ['what is 2 + 2', 'what is 2 plus 2', 'two plus two']:
            return "Two plus two equals four"
        
        return None
    
    def test_voices(self):
        """Test all available voice methods"""
        test_text = "Hello, this is a voice test. How do I sound?"
        
        for voice_method in self.available_tts:
            print(f"\nTesting {voice_method}...")
            original_voice = self.current_tts
            self.current_tts = voice_method
            self.speak(test_text)
            time.sleep(2)  # Pause between tests
            self.current_tts = original_voice
    
    def run(self):
        """Main conversation loop"""
        print("\n" + "="*60)
        print("ENHANCED VOICE ASSISTANT READY")
        print("="*60)
        print("Commands:")
        print("- Press Enter and speak")
        print("- 'test voices' - test all available voice options")
        print("- 'change voice [type]' - switch TTS method")
        print("- 'list voices' - show available voices")
        print("- 'quit' or 'exit' to stop")
        print("="*60)
        
        # Initial greeting
        self.speak(f"Hey, whats up! Ask me anything. The odds are I might crash trying to answer it, but sometimes I don't, so try your luck!")
        
        try:
            while True:
                input("\nPress Enter to start listening... ")
                
                user_input = self.listen()
                if user_input is None:
                    continue
                
                # Handle quit commands
                if any(word in user_input for word in ['quit', 'exit', 'goodbye']):
                    self.speak("Goodbye!")
                    break
                
                # Handle test voices
                if 'test voices' in user_input:
                    self.speak("Testing all available voices")
                    self.test_voices()
                    continue
                
                # Try simple commands first
                simple_response = self.handle_simple_commands(user_input)
                if simple_response is True:  # Voice command handled
                    continue
                elif simple_response:  # Got a response
                    self.speak(simple_response)
                    continue
                
                # Generate AI response
                print("Thinking...")
                response = self.generate_response(user_input)
                self.speak(response)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.speak("Goodbye!")

def main():
    # Install additional TTS tools
    print("Checking for additional TTS tools...")
    print("To get better voices, you can install:")
    print("1. Festival: sudo apt install festival")
    print("2. Audio players: sudo apt install mpg123 mpv")
    print("3. For Google TTS, internet connection is required")
    print()
    
    try:
        assistant = EnhancedVoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
