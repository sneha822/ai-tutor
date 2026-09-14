#!/bin/zsh
# Double-click (or `open` this file) to check global hotkeys from Terminal.app.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/hotkey_check.py
echo
echo "Done. Report saved to logs/hotkey_check.txt. You can close this window."
