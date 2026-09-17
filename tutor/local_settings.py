"""
Per-computer choices the app saves in local_settings.json (never committed): the microphone and speaker picked in
the page, and the camera method that works on this machine.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import config

log = logging.getLogger("settings")

PATH = Path(__file__).resolve().parent.parent / config.LOCAL_SETTINGS_FILE
_lock = threading.Lock()


def load() -> dict:
    try:
        data = json.loads(PATH.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("ignoring unreadable %s (%s)", PATH.name, e)
        return {}


def update(**values) -> None:
    """Merge values into the file. Never raises."""
    with _lock:
        data = load()
        data.update(values)
        try:
            PATH.write_text(json.dumps(data, indent=2) + "\n")
        except Exception:
            log.exception("could not save %s (continuing)", PATH.name)
