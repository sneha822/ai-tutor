#!/bin/zsh
# Double-click (or `open` this file) to run the voice tutor in Terminal.app, which can be granted mic access.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/voice_chat.py
echo
echo "Voice chat ended. Log saved to logs/voice.log. You can close this window."
