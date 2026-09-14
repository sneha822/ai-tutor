"""
Focus interventions (stage 4).

When the focus detector reports sustained distraction and the conversation is idle, the tutor receives a hidden
[SYSTEM: ...] message and briefly, warmly redirects the student.

Demo overrides: force() fires one immediately (even with no working camera); toggle_suppress() blocks the
automatic ones.
"""
from __future__ import annotations

import logging
import threading
import time

import config
from tutor import prompts
from tutor.state import SharedState

log = logging.getLogger("interv")

CHECK_INTERVAL_S = 0.2


class InterventionController:
    def __init__(self, state: SharedState, detector, loop, tutor):
        self._state = state
        self._detector = detector    # FocusDetector, or None if the camera stack failed to start
        self._loop = loop
        self._tutor = tutor
        self._last_fire = float("-inf")
        self._blocked_kind: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="interventions", daemon=True)
        state.interventions_suppressed = not config.INTERVENTIONS_ENABLED

    def start(self) -> None:
        if self._detector is None:
            log.error("NO FOCUS DETECTOR: automatic interventions disabled; the force hotkey still works")
            return
        self._thread.start()
        log.info("focus interventions %s (cooldown %ss)",
                 "SUPPRESSED" if self._state.interventions_suppressed else "armed", config.INTERVENTION_COOLDOWN_S)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def force(self) -> None:
        snap = self._snapshot()
        description = snap.describe() if snap is not None and snap.distracted else config.FORCE_INTERVENTION_REASON
        log.warning("FORCED INTERVENTION: student %s", description)
        self._fire(description, force=True)

    def toggle_suppress(self) -> None:
        self._state.interventions_suppressed = not self._state.interventions_suppressed
        log.warning("AUTOMATIC INTERVENTIONS %s", "SUPPRESSED" if self._state.interventions_suppressed else "ENABLED")

    # ------------------------------------------------------------------- internal

    def _snapshot(self):
        if self._detector is None:
            return None
        try:
            return self._detector.snapshot()
        except Exception:
            log.exception("focus snapshot failed")
            return None

    def _fire(self, description: str, force: bool) -> None:
        self._last_fire = time.monotonic()
        self._loop.intervene(prompts.intervention_message(description, self._tutor.current_topic), force=force)

    def _blocked(self, now: float) -> str | None:
        if self._state.interventions_suppressed:
            return "suppressed by hotkey"
        left = config.INTERVENTION_COOLDOWN_S - (now - self._last_fire)
        if left > 0:
            return f"cooldown, {left:.0f}s left"
        if not self._loop.is_idle():
            return "waiting for the conversation to go quiet"
        return None

    def _run(self) -> None:
        while not self._stop.wait(CHECK_INTERVAL_S):
            try:
                snap = self._snapshot()
                if snap is None or not snap.distracted or not self._state.camera_enabled:
                    self._blocked_kind = None
                    continue
                blocked = self._blocked(time.monotonic())
                if blocked:
                    kind = blocked.split(",")[0]
                    if kind != self._blocked_kind:   # log each reason once, not 5x a second
                        log.info("student %s; not intervening yet: %s", snap.describe(), blocked)
                        self._blocked_kind = kind
                    continue
                self._blocked_kind = None
                log.warning("INTERVENTION: student %s", snap.describe())
                self._fire(snap.describe(), force=False)
            except Exception:
                log.exception("intervention check failed (continuing)")
