"""
Stage 3 check: talk to the tutor. Mic -> VAD -> STT -> tutor -> voice -> output device.

    .venv/bin/python scripts/voice_chat.py        (or open scripts/run_voice_chat.command)

Type in this terminal:  i + Enter = interrupt   q + Enter = quit
Global hotkey: config.HOTKEY_INTERRUPT. Logs also go to logs/voice.log.
"""
import os
import sys
from pathlib import Path

# Every model is cached by scripts/download_models.py; never let venue wifi stall a model load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging  # noqa: E402

import config  # noqa: E402
from tutor import logsetup  # noqa: E402

log = logging.getLogger("app")


def print_event(kind: str, text: str) -> None:
    if kind == "user":
        print(f"\n   you> {text}", flush=True)
    elif kind == "tutor_chunk":
        print(f" tutor> {text}", flush=True)
    elif kind == "interrupted":
        print(f"   [interrupted: {text}]", flush=True)


def main() -> None:
    logsetup.setup("voice.log")
    from tutor import perf
    perf.limit_threads()
    from tutor.audio.devices import load_saved
    from tutor.audio.mic import MicListener
    from tutor.audio.speaker import Speaker
    from tutor.audio.stt import Transcriber
    from tutor.audio.tts import TTS
    from tutor.hotkeys import Hotkeys
    from tutor.state import SharedState
    from tutor.tutor import Tutor
    from tutor.voice import VoiceLoop

    state = SharedState()
    tutor = Tutor()
    stt = Transcriber()
    tts = TTS()
    input_device, output_device = load_saved()   # last choice from the page's device menu, else config.py
    speaker = Speaker(device=output_device)
    loop = VoiceLoop(state, tutor, stt, tts, speaker, on_event=print_event)
    mic = MicListener(state, on_utterance=loop.on_utterance, on_user_speaking=loop.on_user_speaking,
                      on_barge_in=lambda: loop.interrupt("barge-in"), device=input_device)
    hotkeys = Hotkeys({config.HOTKEY_INTERRUPT: lambda: loop.interrupt("hotkey")})

    loop.start()
    mic.start()
    hotkeys.start()
    log.info("READY. Speak to the tutor. (i+Enter interrupt, q+Enter quit, hotkey %s)", config.HOTKEY_INTERRUPT)
    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            if cmd == "i":
                loop.interrupt("terminal")
    except KeyboardInterrupt:
        pass
    finally:
        hotkeys.stop()
        mic.stop()
        loop.stop()
        speaker.close()
        log.info("bye")


if __name__ == "__main__":
    main()
