#!/usr/bin/env python3
"""
Voice-Activated AI Assistant for NVIDIA Jetson Nano 2GB
Using TinyLlama via subprocess calls to avoid memory issues
Optimized for USB Audio Device (hw:2,0) over SSH
"""

import speech_recognition as sr
import subprocess
import sys
import os
import tempfile
import json
import time

class TinyLlamaVoiceAssistant:
    def __init__(self):
        print("Initializing TinyLlama Voice Assistant...")
        
        # Audio device configuration for USB Audio Device
        self.audio_device = "plughw:2,0"  # Your USB audio device
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=11)  # USB Audio Device input
        
        # TinyLlama configuration
        self.model_path = None
        self.llama_executable = None
        self.setup_tinyllama()
        
        print("TinyLlama Voice Assistant initialized successfully!")
        print(f"Audio output: {self.audio_device}")
        print(f"Audio input: Microphone device index 11")
        print(f"AI Model: TinyLlama via {self.llama_executable}")
        
    def setup_tinyllama(self):
        """Setup TinyLlama model and executable"""
        # Check for Ollama first (easiest option)
        if os.path.exists('/bin/ollama'):
            self.llama_executable = '/bin/ollama'
            print("Found Ollama installation")
            return
        
        # Check for llama.cpp
        possible_paths = [
            './llama.cpp/main',
            './main',
            '/usr/local/bin/llama-cpp',
            os.path.expanduser('~/llama.cpp/main')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.llama_executable = path
                print(f"Found llama.cpp at: {path}")
                break
        
        if not self.llama_executable:
            print("No compatible LLM executable found!")
            print("Please install either:")
            print("1. Ollama: wget https://github.com/ollama/ollama/releases/latest/download/ollama-linux-arm64 -O /bin/ollama && chmod +x /bin/ollama")
            print("2. llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j4")
            sys.exit(1)
        
        # Check for model file
        model_paths = [
            'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
            './models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
            os.path.expanduser('~/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                self.model_path = path
                print(f"Found model at: {path}")
                break
    
    def speak(self, text):
        """Text-to-speech using espeak piped to aplay for USB audio"""
        try:
            # Clean text for espeak (remove quotes and special chars)
            clean_text = text.replace('"', '').replace("'", "").replace('`', '').strip()
            if not clean_text:
                return
                
            # Use the method that works: espeak stdout piped to aplay
            cmd = f'espeak -a 200 -s 150 -v en-us "{clean_text}" --stdout | aplay -D {self.audio_device}'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Assistant: {text}")
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"Assistant: {text}")
    
    def listen(self):
        """Listen for voice input"""
        try:
            print("Listening...")
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            print("Processing speech...")
            # Use Google's free speech recognition
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def generate_response_ollama(self, user_input):
        """Generate AI response using Ollama"""
        try:
            # Create conversation prompt
            prompt = f"You are a helpful assistant. Answer briefly and clearly.\n\nHuman: {user_input}\nAssistant:"
            
            # Use Ollama to generate response
            cmd = ['/bin/ollama', 'run', 'tinyllama', prompt]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Clean up response
                if response:
                    # Remove the prompt echo if present
                    if "Assistant:" in response:
                        response = response.split("Assistant:")[-1].strip()
                    return response[:200]  # Limit length
                else:
                    return "I'm not sure how to respond to that."
            else:
                print(f"Ollama error: {result.stderr}")
                return "I'm having trouble processing that request."
                
        except subprocess.TimeoutExpired:
            return "I'm thinking too hard. Could you try again?"
        except Exception as e:
            print(f"AI generation error: {e}")
            return "I'm having trouble processing that request."
    
    def generate_response_llamacpp(self, user_input):
        """Generate AI response using llama.cpp"""
        try:
            if not self.model_path:
                return "I need a model file to generate responses."
            
            # Create conversation prompt
            prompt = f"### Human: {user_input}\n### Assistant:"
            
            # Use llama.cpp to generate response
            cmd = [
                self.llama_executable,
                '-m', self.model_path,
                '-p', prompt,
                '-n', '50',  # Max tokens
                '--temp', '0.7',
                '--top-p', '0.9',
                '--repeat-penalty', '1.1',
                '--ctx-size', '512',  # Small context for memory
                '--threads', '4'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=20  # 20 second timeout
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # Extract just the assistant's response
                if "### Assistant:" in response:
                    response = response.split("### Assistant:")[-1].strip()
                # Clean up response
                lines = response.split('\n')
                clean_response = ' '.join(line.strip() for line in lines if line.strip())
                return clean_response[:200] if clean_response else "I'm not sure how to respond to that."
            else:
                print(f"llama.cpp error: {result.stderr}")
                return "I'm having trouble processing that request."
                
        except subprocess.TimeoutExpired:
            return "I'm thinking too hard. Could you try again?"
        except Exception as e:
            print(f"AI generation error: {e}")
            return "I'm having trouble processing that request."
    
    def generate_response(self, user_input):
        """Generate AI response using available method"""
        if self.llama_executable == '/bin/ollama':
            return self.generate_response_ollama(user_input)
        else:
            return self.generate_response_llamacpp(user_input)
    
    def handle_simple_commands(self, user_input):
        """Handle simple commands without AI"""
        user_input = user_input.lower().strip()
        
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
        
        # Weather (placeholder)
        if 'weather' in user_input:
            return "I don't have access to weather data, but I hope it's nice outside!"
        
        # System info
        if any(word in user_input for word in ['system', 'memory', 'cpu']):
            try:
                # Get basic system info
                with open('/proc/meminfo', 'r') as f:
                    mem_info = f.read()
                mem_total = [line for line in mem_info.split('\n') if 'MemTotal' in line][0]
                mem_available = [line for line in mem_info.split('\n') if 'MemAvailable' in line][0]
                return f"System running on Jetson Nano. {mem_total.split()[1]} KB total memory, {mem_available.split()[1]} KB available."
            except:
                return "I'm running on a Jetson Nano 2GB with ARM64 architecture."
        
        return None  # Use AI for other responses
    
    def test_audio(self):
        """Test audio output"""
        print("Testing audio output...")
        self.speak("Audio test successful. I can speak through your USB headphones.")
    
    def test_ai(self):
        """Test AI model"""
        print("Testing AI model...")
        response = self.generate_response("Hello, can you hear me?")
        print(f"AI Test Response: {response}")
        self.speak(response)
    
    def run(self):
        """Main conversation loop"""
        print("\n" + "="*50)
        print("TINYLLAMA VOICE ASSISTANT READY")
        print("="*50)
        print("Commands:")
        print("- Press Enter and speak after the beep")
        print("- Say 'test audio' to test speakers")
        print("- Say 'test ai' to test the AI model")
        print("- Say 'quit' or 'exit' to stop")
        print("- Ctrl+C to force quit")
        print("="*50)
        
        # Initial greeting
        self.speak("Hello! I'm your voice assistant powered by TinyLlama. Press Enter to talk to me.")
        
        try:
            while True:
                # Wait for Enter key
                input("\nPress Enter to start listening... ")
                
                # Listen for voice input
                user_input = self.listen()
                
                if user_input is None:
                    continue
                
                # Handle special commands
                if any(word in user_input for word in ['quit', 'exit', 'goodbye']):
                    self.speak("Goodbye!")
                    break
                elif 'test audio' in user_input:
                    self.test_audio()
                    continue
                elif 'test ai' in user_input:
                    self.test_ai()
                    continue
                
                # Try simple commands first (faster)
                simple_response = self.handle_simple_commands(user_input)
                if simple_response:
                    print(f"Assistant: {simple_response}")
                    self.speak(simple_response)
                    continue
                
                # Generate AI response for complex queries
                print("Thinking...")
                response = self.generate_response(user_input)
                print(f"Assistant: {response}")
                self.speak(response)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.speak("Goodbye!")
        except Exception as e:
            print(f"Error: {e}")

def main():
    try:
        assistant = TinyLlamaVoiceAssistant()
        assistant.run()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
