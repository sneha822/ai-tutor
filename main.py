"""
AI tutor: voice conversation + webcam focus interventions + browser UI.

    .venv/bin/python main.py        (or open scripts/run_tutor.command, so macOS grants camera + mic to Terminal)

The page opens at http://127.0.0.1:8765 (config.UI_PORT).
Fill in the page's welcome form (name + subject or talk mode) to start; nothing is answered before that.
Type in this terminal (then Enter):
    i interrupt   f force intervention   s toggle suppress   c recalibrate   m mic on/off   v camera on/off
    g start a tutor session without the page   q quit
Global hotkeys: HOTKEY_* in config.py. Logs also go to logs/tutor.log.
"""
import json
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
                 "m mic on/off | v camera on/off | g tutor session without the page | q quit")


def print_event(kind: str, text: str) -> None:
    if kind == "user":
        print(f"\n   you> {text}", flush=True)
    elif kind == "tutor_chunk":
        print(f" tutor> {text}", flush=True)
    elif kind == "emotion":
        print(f"   ({text})", flush=True)
    elif kind == "think":
        step = json.loads(text)
        if step.get("kind") == "step" and step.get("status") != "running":
            print(f"   · {step['label']}{'' if step['status'] == 'done' else ' (failed)'}", flush=True)
    elif kind == "interrupted":
        print(f"   [interrupted: {text}]", flush=True)
    elif kind in ("intervention", "action"):
        print(f"\n   [{text}]", flush=True)
    elif kind == "session":
        info = json.loads(text)
        print(f"\n   [new session: {info['persona']}{' · ' + info['subject'] if info['subject'] else ''}"
              f"{' with ' + info['name'] if info['name'] else ''}]", flush=True)


def main() -> None:
    logsetup.setup("tutor.log")
    from tutor import perf
    perf.limit_threads()   # before any model loads: stops the voice, STT and face tracking fighting over cores
    from tutor.audio.devices import AudioDevices, load_saved
    from tutor.audio.mic import MicListener
    from tutor.audio.speaker import Speaker
    from tutor.audio.stt import Transcriber
    from tutor.audio.tts import TTS
    from tutor import prompts
    from tutor.hotkeys import Hotkeys
    from tutor.interventions import InterventionController
    from tutor.session import Session
    from tutor.state import SharedState
    from tutor.tutor import Tutor
    from tutor.ui_server import UIServer
    from tutor.voice import VoiceLoop

    state = SharedState()
    detector = None
    try:
        from tutor.focus import FocusDetector
        detector = FocusDetector()
        detector.set_busy_check(lambda: state.turn_active or state.tutor_speaking)
        detector.start()
    except Exception:
        log.exception("FOCUS DETECTOR UNAVAILABLE; continuing voice-only (force hotkey still works)")

    ui = None

    def on_event(kind: str, text: str) -> None:
        print_event(kind, text)
        if ui is not None:
            ui.publish(kind, text)

    def camera_context() -> dict:
        """What the AI may know about the camera right now: on/off, and a few words about what it sees."""
        snap = detector.snapshot() if detector is not None else None
        on = state.camera_enabled and snap is not None and snap.camera_ok
        return {"camera_on": on, "observation": snap.observation if on else None}

    tutor = Tutor(context=camera_context)
    try:
        from tutor.notes import NotesLibrary, NotesTools
        library = NotesLibrary(tutor.retriever, on_change=lambda: ui is not None and ui.notify_notes())
        tutor.notes = NotesTools(library, on_focus=lambda focus, reading: ui is not None and ui.note_event(focus, reading))
    except Exception:
        log.exception("NOTES UNAVAILABLE; continuing without the Notes section")
    tutor.on_think = lambda kind, data: on_event("think", json.dumps({"kind": kind, **data}))
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

    def start_session(msg: dict) -> None:
        """The page's welcome form: fresh conversation in the chosen mode, then a spoken greeting."""
        session = Session.from_request(msg)
        if session is None:
            log.warning("ignoring session request with unknown mode %r", msg.get("mode"))
            return
        camera = msg.get("camera")
        if isinstance(camera, bool) and camera != state.camera_enabled:
            toggle_camera()
        interventions.reset()
        loop.begin_session(session, prompts.greeting_message(session))
        log.info("SESSION: %s (camera %s)", session.label(), "on" if state.camera_enabled else "off")
        on_event("session", json.dumps(session.to_dict()))

    def explain_note(msg: dict) -> None:
        """The Explain button on a note card: the AI opens that note and starts teaching it."""
        note = tutor.notes.library.get(str(msg.get("note_id") or "")) if tutor.notes is not None else None
        section = msg.get("section")
        section = section if isinstance(section, int) and note and 1 <= section <= len(note.sections) else None
        if note is None or note.status != "ready":
            log.warning("Explain ignored: note %r isn't ready", msg.get("note_id"))
        elif state.session is None:
            log.warning("Explain ignored: no session yet")
        else:
            where = f" · from “{note.sections[section - 1]['title']}”" if section and section > 1 else ""
            loop.intervene(prompts.explain_note_message(note.id, note.title, section), force=True,
                           label=f"Explain · {note.title}{where}", kind="action")

    def ask_text(msg: dict) -> None:
        """A message typed in the page, or a suggested question clicked."""
        text = str(msg.get("text") or "").strip()[:2000]
        if text:
            loop.ask(text)

    actions = {
        "i": lambda: loop.interrupt("terminal"),
        "f": interventions.force,
        "s": interventions.toggle_suppress,
        "c": recalibrate,
        "m": toggle_mic,
        "v": toggle_camera,
        "g": lambda: start_session({"mode": "tutor"}),
    }
    hotkeys = Hotkeys({
        config.HOTKEY_INTERRUPT: lambda: loop.interrupt("hotkey"),
        config.HOTKEY_FORCE_INTERVENTION: interventions.force,
        config.HOTKEY_TOGGLE_SUPPRESS: interventions.toggle_suppress,
        config.HOTKEY_RECALIBRATE: recalibrate,
        config.HOTKEY_TOGGLE_MIC: toggle_mic,
        config.HOTKEY_TOGGLE_CAMERA: toggle_camera,
    })
    ui = UIServer(state, detector, devices=devices, notes=tutor.notes,
                  param_actions={"start_session": start_session, "explain_note": explain_note, "ask_text": ask_text},
                  actions={
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
    if ui.started:
        log.info("READY. Fill in the welcome form at %s to start. %s", ui.url, COMMANDS_HELP)
    else:
        log.warning("No page, so starting a general tutor session right away")
        start_session({"mode": "tutor"})
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
