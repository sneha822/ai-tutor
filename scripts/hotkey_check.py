"""
Can global hotkeys work from this terminal? Run it in the terminal app you will demo from:

    .venv/bin/python scripts/hotkey_check.py        (or open scripts/run_hotkey_check.command)

Uses the app's own hotkey code (tutor/hotkeys.py). Reports macOS Accessibility trust, then listens for 20 s
while you press config.HOTKEY_INTERRUPT a few times plus any other keys. Only counts are recorded, never
which keys. Report: logs/hotkey_check.txt
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from tutor.hotkeys import Hotkeys  # noqa: E402

REPORT = ROOT / "logs" / "hotkey_check.txt"
LISTEN_S = 20


def out(line: str) -> None:
    print(line, flush=True)
    with REPORT.open("a") as f:
        f.write(line + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(f"hotkey check {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        from ApplicationServices import AXIsProcessTrusted
        out(f"Accessibility trusted (AXIsProcessTrusted): {bool(AXIsProcessTrusted())}")
    except Exception as e:
        out(f"Accessibility check unavailable: {e}")

    counts = {"keys": 0, "hotkey": 0}

    def on_hotkey():
        counts["hotkey"] += 1
        out("  hotkey fired")

    def on_key():
        counts["keys"] += 1

    hotkeys = Hotkeys({config.HOTKEY_INTERRUPT: on_hotkey}, on_any_key=on_key)
    hotkeys.start()
    out(f"Press {config.HOTKEY_INTERRUPT} a few times, and a few other keys, in the next {LISTEN_S}s...")
    try:
        subprocess.Popen(["say", "Press control, option, I a few times now."])
    except Exception:
        pass
    time.sleep(LISTEN_S)
    hotkeys.stop()

    out(f"RESULT: key events seen={counts['keys']}, hotkey fired={counts['hotkey']}")
    if counts["keys"] == 0:
        out("VERDICT: no key events reached Python -> enable this terminal under Privacy & Security > Input Monitoring.")
    elif counts["hotkey"] == 0:
        out("VERDICT: keys arrive but the combo never matched -> check HOTKEY_INTERRUPT in config.py.")
    else:
        out("VERDICT: global hotkeys work from this terminal.")


if __name__ == "__main__":
    main()
