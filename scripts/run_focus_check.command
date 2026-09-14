#!/bin/zsh
# Double-click (or `open` this file) to run the guided focus check in Terminal.app.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/focus_check.py
echo
echo "Done. Report saved to logs/focus_check.txt. You can close this window."
