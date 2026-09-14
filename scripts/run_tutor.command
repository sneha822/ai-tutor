#!/bin/zsh
# Double-click (or `open` this file) to run the full tutor in Terminal.app, which can be granted camera + mic access.
cd "$(dirname "$0")/.."
.venv/bin/python main.py
echo
echo "Tutor ended. Log saved to logs/tutor.log. You can close this window."
