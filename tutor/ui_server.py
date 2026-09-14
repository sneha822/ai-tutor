"""
Local web UI (stage 5): FastAPI + one WebSocket, served from a background thread inside the app process.

Pushes conversation events (student transcript, streamed tutor text, interruptions, focus nudges) plus a status
update UI_FOCUS_HZ times a second, and receives button clicks (interrupt, force, suppress, recalibrate, mic/camera)
and the microphone/speaker menu's requests. Only scores and text reach the page, never video. Bound to localhost.
A UI failure never stops the tutor.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from tutor.state import SharedState

log = logging.getLogger("ui")

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
HISTORY_EVENTS = 400   # replayed to a page that connects or reloads mid-session
DEVICE_ACTIONS = ("list_devices", "set_input_device", "set_output_device", "test_speaker")
SPEAKER_BUSY = "The tutor is talking right now, so you're already hearing this speaker"


class UIServer:
    def __init__(self, state: SharedState, detector=None, actions: dict[str, Callable[[], None]] | None = None,
                 devices=None, param_actions: dict[str, Callable[[dict], None]] | None = None):
        self._state = state
        self._detector = detector
        self._actions = actions or {}
        self._param_actions = param_actions or {}   # actions that take the page's message, e.g. start_session
        self._devices = devices   # tutor.audio.devices.AudioDevices, for the microphone/speaker menu
        self.started = False
        self._clients: set[WebSocket] = set()
        self._history: deque[dict] = deque(maxlen=HISTORY_EVENTS)
        self._history_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.url = f"http://{config.UI_HOST}:{config.UI_PORT}"
        self.app = self._build_app()

    # ------------------------------------------------------------ called from other threads

    def start(self) -> None:
        if not (UI_DIR / "index.html").exists():
            log.error("UI FILES MISSING at %s; continuing without the page", UI_DIR)
            return
        # log_config=None: uvicorn's default dictConfig closes every existing logging handler, which silently
        # stopped logs/tutor.log mid-session. Its loggers propagate to the app's handlers instead.
        cfg = uvicorn.Config(self.app, host=config.UI_HOST, port=config.UI_PORT, log_config=None,
                             log_level="warning", access_log=False, lifespan="on")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._serve, name="ui", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self._server.started and self._thread.is_alive():
            time.sleep(0.05)
        if not self._server.started:
            log.error("UI SERVER DID NOT START (is port %d in use?); the tutor keeps running without the page",
                      config.UI_PORT)
            return
        self.started = True
        log.info("UI ready at %s", self.url)
        if config.UI_OPEN_BROWSER:
            try:
                webbrowser.open(self.url)
            except Exception:
                log.warning("could not open a browser; open %s manually", self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3)

    def publish(self, kind: str, text: str) -> None:
        """Forward a VoiceLoop event to the page (thread-safe, never raises)."""
        try:
            event = self._event(kind, text)
            if event is None:
                return
            with self._history_lock:
                self._history.append(event)
            self._broadcast_threadsafe(event)
        except Exception:
            log.exception("UI publish failed (continuing)")

    def _broadcast_threadsafe(self, message: dict) -> None:
        loop = self._loop
        if loop is not None and self._clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)

    # ------------------------------------------------------------------- internal

    def _serve(self) -> None:
        try:
            self._server.run()
        except SystemExit:
            log.error("UI SERVER EXITED (is port %d in use?)", config.UI_PORT)
        except Exception:
            log.exception("UI SERVER CRASHED; the tutor keeps running")

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            self._loop = asyncio.get_running_loop()
            pumps = [asyncio.create_task(self._status_pump()), asyncio.create_task(self._level_pump())]
            yield
            for pump in pumps:
                pump.cancel()

        app = FastAPI(title="AI Tutor", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
        app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

        @app.get("/")
        async def index():
            return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})

        @app.websocket("/ws")
        async def ws(socket: WebSocket):
            await socket.accept()
            self._clients.add(socket)
            with self._history_lock:
                backlog = list(self._history)
            try:
                devices = await asyncio.to_thread(self._devices_snapshot)
                session = self._state.session
                await socket.send_text(json.dumps({"type": "hello", "events": backlog, "status": self._status(),
                                                   "privacy": self._privacy(), "devices": devices,
                                                   "session": session.to_dict() if session else None}))
                while True:
                    raw = await socket.receive_text()
                    # In a worker thread: switching an audio device takes a moment and must not stall the pumps.
                    await asyncio.to_thread(self._handle_client_message, raw)
            except WebSocketDisconnect:
                pass
            except Exception:
                log.exception("websocket error (client dropped)")
            finally:
                self._clients.discard(socket)

        return app

    def _handle_client_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            action = msg.get("action")
        except Exception:
            log.warning("ignoring malformed UI message %r", raw[:80])
            return
        if action in DEVICE_ACTIONS:
            self._device_request(action, msg)
            return
        if action in self._param_actions:
            log.info("UI %s", action)
            try:
                self._param_actions[action](msg)
            except Exception:
                log.exception("UI action %s failed (continuing)", action)
            return
        fn = self._actions.get(action)
        if fn is None:
            log.warning("unknown UI action %r", action)
            return
        log.info("UI button %s", action)
        try:
            fn()
        except Exception:
            log.exception("UI action %s failed (continuing)", action)

    def _device_request(self, action: str, msg: dict) -> None:
        """Microphone/speaker menu: list devices, switch one, or play a test sound. Replies go to every open page."""
        if self._devices is None:
            log.warning("UI %s ignored: audio devices unavailable", action)
            return
        try:
            if action == "test_speaker":
                log.info("UI test_speaker")
                if not self._devices.test_speaker():
                    self._broadcast_threadsafe({"type": "notice", "text": SPEAKER_BUSY})
                return
            if action in ("set_input_device", "set_output_device"):
                name = msg.get("device")
                name = name[:200] if isinstance(name, str) and name else None
                log.info("UI %s %r", action, name or "system default")
                if action == "set_input_device":
                    self._devices.set_input(name)
                else:
                    self._devices.set_output(name)
            self._broadcast_threadsafe(self._devices.snapshot(rescan=action == "list_devices"))
        except Exception:
            log.exception("UI %s failed (continuing)", action)

    def _devices_snapshot(self) -> dict | None:
        if self._devices is None:
            return None
        try:
            return self._devices.snapshot()
        except Exception:
            log.exception("audio device list for UI failed")
            return None

    @staticmethod
    def _privacy() -> dict:
        """Facts for the page's privacy panel, taken from the live configuration."""
        return {"llm_model": config.LLM_MODEL, "llm_fallback": config.LLM_FALLBACK_MODEL,
                "stt_model": config.STT_GROQ_MODEL, "stt_local": f"faster-whisper {config.STT_LOCAL_MODEL}",
                "history_turns": config.HISTORY_TURNS, "tts": f"Kokoro-82M, voice {config.TTS_VOICE}"}

    @staticmethod
    def _event(kind: str, text: str) -> dict | None:
        t = time.time()
        if kind in ("user", "tutor_chunk"):
            return {"type": kind, "text": text, "t": t}
        if kind == "tutor_done":
            return {"type": "tutor_done", "t": t}
        if kind == "interrupted":
            return {"type": "interrupted", "source": text, "t": t}
        if kind == "intervention":   # only the label reaches the page, never the hidden [SYSTEM: ...] text
            return {"type": "nudge", "reason": text, "t": t}
        if kind == "emotion":
            return {"type": "emotion", "emotion": text, "t": t}
        if kind == "session":
            return {"type": "session", "session": json.loads(text), "t": t}
        return None

    def _emotion(self) -> str:
        """The AI's emotion, relaxing to neutral EMOTION_HOLD_S after it last spoke."""
        s = self._state
        if s.turn_active or s.tutor_speaking:
            return s.emotion
        quiet_for = time.monotonic() - max(s.emotion_changed_at, s.playback_ended_at)
        return s.emotion if quiet_for < config.EMOTION_HOLD_S else "neutral"

    def _status(self) -> dict:
        s = self._state
        session = s.session
        status = {"type": "status", "tutor_speaking": s.tutor_speaking, "user_speaking": s.user_speaking,
                  "thinking": s.turn_active and not s.tutor_speaking, "suppressed": s.interventions_suppressed,
                  "camera_enabled": s.camera_enabled, "mic_enabled": s.mic_enabled,
                  "stt_source": s.last_stt_source, "threshold": config.FOCUS_THRESHOLD, "focus": None,
                  "session": session.to_dict() if session else None, "emotion": self._emotion(),
                  "strikes": s.distraction_strikes,
                  "focus_tracking": bool(session and session.tutoring and s.camera_enabled)}
        if self._detector is not None:
            try:
                snap = self._detector.snapshot()
                status["focus"] = {"score": snap.score, "state": snap.state, "reason": snap.reason,
                                   "distracted": snap.distracted, "face": snap.face_present,
                                   "camera_ok": snap.camera_ok, "calibrated": snap.calibrated,
                                   "expression": snap.expression, "observation": snap.observation,
                                   "description": snap.describe() if snap.state in ("UNFOCUSED", "DISTRACTED") else ""}
            except Exception:
                log.exception("focus snapshot for UI failed")
        return status

    async def _broadcast(self, message: dict) -> None:
        data = json.dumps(message)
        for socket in list(self._clients):
            try:
                await socket.send_text(data)
            except Exception:
                self._clients.discard(socket)

    async def _status_pump(self) -> None:
        period = 1.0 / max(1, config.UI_FOCUS_HZ)
        while True:
            await asyncio.sleep(period)
            if self._clients:
                try:
                    await self._broadcast(self._status())
                except Exception:
                    log.exception("UI status update failed (continuing)")

    async def _level_pump(self) -> None:
        """Voice loudness for the avatar (tutor playback + mic), sent UI_AUDIO_HZ times a second."""
        period = 1.0 / max(1, config.UI_AUDIO_HZ)
        last = None
        while True:
            await asyncio.sleep(period)
            if not self._clients:
                continue
            levels = (round(self._state.tutor_level, 3), round(self._state.user_level, 3))
            if levels == last and levels == (0.0, 0.0):
                continue   # don't stream silence
            last = levels
            try:
                await self._broadcast({"type": "level", "tutor": levels[0], "user": levels[1]})
            except Exception:
                log.exception("UI level update failed (continuing)")
