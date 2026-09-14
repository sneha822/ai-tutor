"""
Audio devices: lookup by name, the lists shown in the page's microphone/speaker menu, and live switching.

Devices are matched by name rather than index, so plugging something in can't silently switch them. A choice made in
the page is saved to LOCAL_SETTINGS_FILE (on this computer only, not committed) and used again on the next start,
ahead of INPUT_DEVICE / OUTPUT_DEVICE in config.py.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import sounddevice as sd

import config

if TYPE_CHECKING:
    from tutor.audio.mic import MicListener
    from tutor.audio.speaker import Speaker
    from tutor.state import SharedState

log = logging.getLogger("audio")

SETTINGS_PATH = Path(__file__).resolve().parents[2] / config.LOCAL_SETTINGS_FILE
_CHANNELS = {"input": "max_input_channels", "output": "max_output_channels"}


def _default_hostapi() -> int | None:
    try:
        return int(sd.default.hostapi)
    except Exception:
        return None


def find_device(name: str | None, kind: str) -> int | None:
    """Index of the device called `name` (exact match first, then partial) that supports `kind` ("input"/"output").

    Prefers the system's default audio API, because Windows lists the same device once per API.
    """
    if not name:
        return None
    key = _CHANNELS[kind]
    try:
        devices = sd.query_devices()
    except Exception as e:
        log.error("Could not list audio devices (%s); using system default %s", e, kind)
        return None
    wanted = name.lower()
    api = _default_hostapi()
    matches = [i for i, d in enumerate(devices) if d[key] > 0 and wanted in d["name"].lower()]
    if matches:
        return min(matches, key=lambda i: (devices[i]["name"].lower() != wanted, devices[i]["hostapi"] != api))
    log.error("%s DEVICE %r NOT FOUND; using system default. Available: %s", kind.upper(), name, list_devices(kind))
    return None


def list_devices(kind: str) -> list[str]:
    """Names of the devices that can record ("input") or play ("output"), once each, in the system's order."""
    key = _CHANNELS[kind]
    try:
        devices = sd.query_devices()
    except Exception as e:
        log.error("Could not list audio devices (%s)", e)
        return []
    api = _default_hostapi()
    names: list[str] = []
    for d in devices:
        if d[key] > 0 and (api is None or d["hostapi"] == api) and d["name"] not in names:
            names.append(d["name"])
    return names


def default_device_name(kind: str) -> str | None:
    try:
        return sd.query_devices(kind=kind)["name"]
    except Exception:
        return None


def load_saved() -> tuple[str | None, str | None]:
    """(input, output) device names to start with: the page's last choice if there is one, else config.py."""
    names = {"input": config.INPUT_DEVICE, "output": config.OUTPUT_DEVICE}
    try:
        saved = json.loads(SETTINGS_PATH.read_text())
    except FileNotFoundError:
        saved = {}
    except Exception as e:
        log.warning("ignoring unreadable %s (%s)", SETTINGS_PATH.name, e)
        saved = {}
    if isinstance(saved, dict):
        for kind in names:
            if f"{kind}_device" in saved:
                names[kind] = saved[f"{kind}_device"] or None
    return names["input"], names["output"]


class AudioDevices:
    """Backs the page's microphone/speaker menu: lists devices, switches them live and remembers the choice."""

    def __init__(self, state: SharedState, mic: MicListener, speaker: Speaker):
        self._state = state
        self._mic = mic
        self._speaker = speaker
        self._lock = threading.Lock()

    def snapshot(self, rescan: bool = False) -> dict:
        """Everything the menu shows. rescan=True first looks for devices plugged in since the last look."""
        with self._lock:
            if rescan:
                self._rescan()
            return {"type": "devices",
                    "inputs": list_devices("input"), "outputs": list_devices("output"),
                    "input": self._mic.device, "output": self._speaker.device,
                    "input_active": self._mic.active_device, "output_active": self._speaker.active_device,
                    "default_input": default_device_name("input"), "default_output": default_device_name("output")}

    def set_input(self, name: str | None) -> None:
        with self._lock:
            if name is not None and name not in list_devices("input"):
                log.warning("ignoring unknown microphone %r", name)
                return
            self._mic.set_device(name)
            self._save()

    def set_output(self, name: str | None) -> None:
        with self._lock:
            if name is not None and name not in list_devices("output"):
                log.warning("ignoring unknown speaker %r", name)
                return
            self._speaker.set_device(name)
            self._save()

    def test_speaker(self) -> bool:
        return self._speaker.play_test()

    def _rescan(self) -> None:
        """PortAudio only lists devices when it initializes, so re-initialize it with both streams closed.
        Skipped mid-conversation, where reopening the devices would cut off speech."""
        s = self._state
        if s.turn_active or s.tutor_speaking or s.user_speaking:
            return
        self._mic.close_stream()
        self._speaker.close_stream()
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            log.exception("audio device rescan failed (continuing with the old list)")
        self._speaker.reopen()
        self._mic.reopen()

    def _save(self) -> None:
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
            if not isinstance(saved, dict):
                saved = {}
        except Exception:
            saved = {}
        saved.update(input_device=self._mic.device, output_device=self._speaker.device)
        try:
            SETTINGS_PATH.write_text(json.dumps(saved, indent=2) + "\n")
        except Exception:
            log.exception("could not save the device choice to %s (continuing)", SETTINGS_PATH.name)
