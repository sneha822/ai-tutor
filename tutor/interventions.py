"""
Focus interventions (stage 4). Tutor mode only, and only while the camera is on.

When the focus detector reports sustained distraction and the conversation is idle, the tutor receives a hidden
[SYSTEM: ...] message and redirects the student. Distractions close together escalate: the first is light, the
second annoyed, the third and later properly angry (stern but caring).

The tutor also checks in when the student has looked puzzled for a while right after an explanation, or keeps
yawning.

Demo overrides: force() fires a distraction nudge immediately (even with no working camera); toggle_suppress()
blocks the automatic ones.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

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
        self._last_fire = float("-inf")      # any proactive message: nudge or check-in
        self._last_checkin = float("-inf")
        self._strikes: deque[float] = deque()
        self._puzzled_since: float | None = None
        self._yawns_handled = 0
        self._blocked_kind: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="interventions", daemon=True)
        state.interventions_suppressed = not config.INTERVENTIONS_ENABLED

    def start(self) -> None:
        if self._detector is None:
            log.error("NO FOCUS DETECTOR: automatic interventions disabled; the force hotkey still works")
            return
        self._thread.start()
        log.info("focus interventions %s for tutor sessions (cooldown %ss, escalation window %ss)",
                 "SUPPRESSED" if self._state.interventions_suppressed else "armed",
                 config.INTERVENTION_COOLDOWN_S, config.STRIKE_WINDOW_S)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def reset(self) -> None:
        """A new session: forget earlier distractions and check-ins."""
        self._strikes.clear()
        self._state.distraction_strikes = 0
        self._last_fire = self._last_checkin = float("-inf")
        self._puzzled_since = None
        self._yawns_handled = 0

    def force(self) -> None:
        if not self._tutoring():
            log.warning("FORCED INTERVENTION ignored: focus nudges only happen in tutor sessions")
            return
        snap = self._snapshot()
        description = snap.describe() if snap is not None and snap.distracted else config.FORCE_INTERVENTION_REASON
        log.warning("FORCED INTERVENTION: student %s", description)
        self._nudge(description, force=True)

    def toggle_suppress(self) -> None:
        self._state.interventions_suppressed = not self._state.interventions_suppressed
        log.warning("AUTOMATIC INTERVENTIONS %s", "SUPPRESSED" if self._state.interventions_suppressed else "ENABLED")

    # ------------------------------------------------------------------- internal

    def _tutoring(self) -> bool:
        session = self._state.session
        return session is not None and session.tutoring

    def _snapshot(self):
        if self._detector is None:
            return None
        try:
            return self._detector.snapshot()
        except Exception:
            log.exception("focus snapshot failed")
            return None

    def _strike_count(self, now: float) -> int:
        while self._strikes and now - self._strikes[0] > config.STRIKE_WINDOW_S:
            self._strikes.popleft()
        self._state.distraction_strikes = len(self._strikes)
        return len(self._strikes)

    def _nudge(self, description: str, force: bool) -> None:
        now = time.monotonic()
        self._strikes.append(now)
        count = self._strike_count(now)
        self._last_fire = now
        window_min = round(config.STRIKE_WINDOW_S / 60)
        label = f"Focus nudge · {description}"
        if count > 1:
            label += f" · {prompts.ordinal(count)} time in {window_min} min"
        log.warning("focus nudge #%d in the last %d min", count, window_min)
        message = prompts.intervention_message(description, self._tutor.current_topic, count, window_min)
        self._loop.intervene(message, force=force, label=label)

    def _checkin(self, kind: str, count: int = 0) -> None:
        now = time.monotonic()
        self._last_fire = self._last_checkin = now
        label = "Check-in · you looked puzzled" if kind == "puzzled" else f"Check-in · {count} yawns in a few minutes"
        log.warning("CHECK-IN: %s", label)
        self._loop.intervene(prompts.checkin_message(kind, self._tutor.current_topic, count), label=label)

    def _blocked(self, now: float) -> str | None:
        if self._state.interventions_suppressed:
            return "suppressed by hotkey"
        left = config.INTERVENTION_COOLDOWN_S - (now - self._last_fire)
        if left > 0:
            return f"cooldown, {left:.0f}s left"
        if not self._loop.is_idle():
            return "waiting for the conversation to go quiet"
        return None

    def _check_puzzled(self, now: float, snap) -> None:
        if snap.expression != "frowning" or snap.state != "FOCUSED":
            self._puzzled_since = None
            return
        self._puzzled_since = self._puzzled_since or now
        explained = (len(self._tutor.history) >= 4   # at least one real exchange after the greeting
                     and now - self._state.playback_ended_at <= config.PUZZLED_AFTER_SPEECH_S)
        if (now - self._puzzled_since >= config.PUZZLED_CHECKIN_S and explained
                and now - self._last_checkin >= config.CHECKIN_COOLDOWN_S and self._blocked(now) is None):
            self._puzzled_since = None
            self._checkin("puzzled")

    def _check_tired(self, now: float, snap) -> None:
        self._yawns_handled = min(self._yawns_handled, snap.yawns_recent)   # old yawns left the window
        if (snap.yawns_recent - self._yawns_handled >= config.YAWNS_FOR_CHECKIN
                and now - self._last_checkin >= config.CHECKIN_COOLDOWN_S and self._blocked(now) is None):
            self._yawns_handled = snap.yawns_recent
            self._checkin("tired", snap.yawns_recent)

    def _run(self) -> None:
        while not self._stop.wait(CHECK_INTERVAL_S):
            try:
                now = time.monotonic()
                self._strike_count(now)
                snap = self._snapshot()
                if snap is None or not snap.camera_ok or not self._tutoring() or not self._state.camera_enabled:
                    self._blocked_kind = None
                    self._puzzled_since = None
                    continue
                if not snap.distracted:
                    self._blocked_kind = None
                    self._check_puzzled(now, snap)
                    self._check_tired(now, snap)
                    continue
                self._puzzled_since = None
                blocked = self._blocked(now)
                if blocked:
                    kind = blocked.split(",")[0]
                    if kind != self._blocked_kind:   # log each reason once, not 5x a second
                        log.info("student %s; not intervening yet: %s", snap.describe(), blocked)
                        self._blocked_kind = kind
                    continue
                self._blocked_kind = None
                log.warning("INTERVENTION: student %s", snap.describe())
                self._nudge(snap.describe(), force=False)
            except Exception:
                log.exception("intervention check failed (continuing)")
