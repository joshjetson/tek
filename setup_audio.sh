#!/bin/bash
# Audio setup script for USB Audio Device on Jetson Nano

echo "Setting up USB Audio Device for Voice Assistant"
echo "=============================================="

# Create proper .asoundrc file
echo "Creating .asoundrc file..."
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type plug
    slave.pcm "hw:2,0"
}
ctl.!default {
    type hw
    card 2
}
EOF

echo "✓ .asoundrc created"

# Set environment variables for ALSA
echo "Setting ALSA environment variables..."
export ALSA_PCM_DEVICE=2
export ALSA_PCM_CARD=2

# Add to .bashrc for persistence
if ! grep -q "ALSA_PCM_DEVICE" ~/.bashrc; then
    echo "export ALSA_PCM_DEVICE=2" >> ~/.bashrc
    echo "export ALSA_PCM_CARD=2" >> ~/.bashrc
    echo "✓ Added ALSA environment variables to .bashrc"
fi

# Test audio output
echo "Testing audio output..."
echo "You should hear 'Audio test successful' through your USB headphones:"
espeak -a 200 -v en-us "Audio test successful" --stdout | aplay -D plughw:2,0

echo ""
echo "Setup complete!"
echo "If you heard the test message, your audio is properly configured."
echo "You can now run the voice assistant with: python3 voice_assistant.py"
