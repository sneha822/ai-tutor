"""
AI tutor: voice conversation + webcam focus interventions + browser UI.

    .venv/bin/python main.py        (or open scripts/run_tutor.command, so macOS grants camera + mic to Terminal)

The page opens at http://127.0.0.1:8765 (config.UI_PORT).
Type in this terminal (then Enter):
    i interrupt   f force intervention   s toggle suppress   c recalibrate   m mic on/off   v camera on/off   q quit
Global hotkeys: HOTKEY_* in config.py. Logs also go to logs/tutor.log.
"""
import os
import sys
from pathlib import Path

# Every model is cached by scripts/download_models.py; never let venue wifi stall a model load.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging  # noqa: E402

import config  # noqa: E402
from tutor import logsetup  # noqa: E402

log = logging.getLogger("app")

COMMANDS_HELP = ("commands: i interrupt | f force intervention | s toggle suppress | c recalibrate | "
                 "m mic on/off | v camera on/off | q quit")


def print_event(kind: str, text: str) -> None:
    if kind == "user":
        print(f"\n   you> {text}", flush=True)
    elif kind == "tutor_chunk":
        print(f" tutor> {text}", flush=True)
    elif kind == "interrupted":
        print(f"   [interrupted: {text}]", flush=True)
    elif kind == "intervention":
        print("\n   [focus intervention]", flush=True)


def main() -> None:
    logsetup.setup("tutor.log")
    from tutor.audio.devices import AudioDevices, load_saved
    from tutor.audio.mic import MicListener
    from tutor.audio.speaker import Speaker
    from tutor.audio.stt import Transcriber
    from tutor.audio.tts import TTS
    from tutor.hotkeys import Hotkeys
    from tutor.interventions import InterventionController
    from tutor.state import SharedState
    from tutor.tutor import Tutor
    from tutor.ui_server import UIServer
    from tutor.voice import VoiceLoop

    state = SharedState()
    detector = None
    try:
        from tutor.focus import FocusDetector
        detector = FocusDetector()
        detector.start()
    except Exception:
        log.exception("FOCUS DETECTOR UNAVAILABLE; continuing voice-only (force hotkey still works)")

    ui = None

    def on_event(kind: str, text: str) -> None:
        print_event(kind, text)
        if ui is not None:
            ui.publish(kind, text)

    tutor = Tutor()
    stt = Transcriber()
    tts = TTS()
    input_device, output_device = load_saved()   # last choice from the page's device menu, else config.py
    speaker = Speaker(state, device=output_device)
    loop = VoiceLoop(state, tutor, stt, tts, speaker, on_event=on_event)
    mic = MicListener(state, on_utterance=loop.on_utterance, on_user_speaking=loop.on_user_speaking,
                      on_barge_in=lambda: loop.interrupt("barge-in"), device=input_device)
    devices = AudioDevices(state, mic, speaker)
    interventions = InterventionController(state, detector, loop, tutor)

    def recalibrate() -> None:
        if detector is None:
            log.error("no focus detector to recalibrate")
        else:
            detector.recalibrate()

    def toggle_mic() -> None:
        state.mic_enabled = not state.mic_enabled
        mic.set_enabled(state.mic_enabled)

    def toggle_camera() -> None:
        state.camera_enabled = not state.camera_enabled
        if detector is None:
            log.error("no focus detector; camera toggle only affects the page's self-view")
        else:
            detector.set_enabled(state.camera_enabled)

    actions = {
        "i": lambda: loop.interrupt("terminal"),
        "f": interventions.force,
        "s": interventions.toggle_suppress,
        "c": recalibrate,
        "m": toggle_mic,
        "v": toggle_camera,
    }
    hotkeys = Hotkeys({
        config.HOTKEY_INTERRUPT: lambda: loop.interrupt("hotkey"),
        config.HOTKEY_FORCE_INTERVENTION: interventions.force,
        config.HOTKEY_TOGGLE_SUPPRESS: interventions.toggle_suppress,
        config.HOTKEY_RECALIBRATE: recalibrate,
        config.HOTKEY_TOGGLE_MIC: toggle_mic,
        config.HOTKEY_TOGGLE_CAMERA: toggle_camera,
    })
    ui = UIServer(state, detector, devices=devices, actions={
        "interrupt": lambda: loop.interrupt("button"),
        "force": interventions.force,
        "toggle_suppress": interventions.toggle_suppress,
        "recalibrate": recalibrate,
        "toggle_mic": toggle_mic,
        "toggle_camera": toggle_camera,
    })

    loop.start()
    mic.start()
    interventions.start()
    hotkeys.start()
    ui.start()
    log.info("READY. Speak to the tutor. %s", COMMANDS_HELP)
    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            if cmd in actions:
                log.info("terminal command %r", cmd)
                actions[cmd]()
            elif cmd:
                print(COMMANDS_HELP, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()
        hotkeys.stop()
        interventions.stop()
        mic.stop()
        loop.stop()
        if detector is not None:
            detector.stop()
        speaker.close()
        log.info("bye")


if __name__ == "__main__":
    main()
