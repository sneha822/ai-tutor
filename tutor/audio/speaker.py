"""
Audio output. Playback reads from an in-memory buffer in the device callback, so stop() silences the
tutor within one 20 ms block and discards everything queued. The callback also measures the loudness of the
exact audio being played, which drives the avatar's mouth in the UI.

The output device can be switched while the app runs (set_device); audio still queued carries on on the new one.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

import config
from tutor.audio.devices import find_device
from tutor.state import SharedState

log = logging.getLogger("speaker")

RATE = 24000           # Kokoro's output rate; audio is resampled only for a device that can't open at it
BLOCK = 480            # 20 ms
LEVEL_GAIN = 4.0       # Kokoro speech RMS ~0.05-0.15 -> level ~0.2-0.6
LEVEL_RELEASE = 0.6    # per block: rises instantly, falls smoothly over ~60 ms


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(audio) == 0:
        return audio
    n = max(1, int(round(len(audio) * dst / src)))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def _chime() -> np.ndarray:
    """Two soft notes for the speaker test in the page."""
    def note(freq: float, seconds: float) -> np.ndarray:
        t = np.arange(int(RATE * seconds)) / RATE
        envelope = np.minimum(1.0, t / 0.01) * np.exp(-t * 6.0)
        return 0.2 * envelope * (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t))
    return np.concatenate([note(659.25, 0.18), note(987.77, 0.45)]).astype(np.float32)


class Speaker:
    def __init__(self, state: SharedState | None = None, device: str | None = config.OUTPUT_DEVICE):
        self._state = state
        self._lock = threading.Lock()          # playback queue, shared with the device callback
        self._stream_lock = threading.Lock()   # opening and closing the device
        self._chunks: deque[np.ndarray] = deque()
        self._pos = 0
        self._level = 0.0
        self._drained = threading.Event()
        self._drained.set()
        self.first_play_t: float | None = None   # when audio first reached the device this turn
        self.device = device                     # chosen output name (None = system default)
        self.active_device = ""                  # name of the device actually open
        self._rate = RATE                        # rate the open device plays at; queued audio is stored at it
        self._stream: sd.OutputStream | None = None
        with self._stream_lock:
            self._open()

    # ------------------------------------------------------------------ device

    def set_device(self, name: str | None) -> None:
        """Switch to another output (None = system default). Queued speech continues on the new device."""
        self.device = name
        log.info("Speaker switching to %s", name or "system default")
        self.reopen()

    def reopen(self) -> None:
        with self._stream_lock:
            self._close_locked()
            self._open()

    def close_stream(self) -> None:
        with self._stream_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        stream, self._stream = self._stream, None
        self.active_device = ""
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _open(self) -> None:
        """Open the chosen device at 24 kHz, or at its native rate if it refuses. Caller holds _stream_lock."""
        index = find_device(self.device, "output")
        try:
            native = int(sd.query_devices(index, "output")["default_samplerate"])
        except Exception:
            native = 48000
        for rate in dict.fromkeys((RATE, native)):
            try:
                stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", blocksize=BLOCK * rate // RATE,
                                         device=index, latency="low", callback=self._callback)
                stream.start()
            except Exception as e:
                log.warning("Speaker open at %d Hz failed: %s", rate, e)
                continue
            self._set_rate(rate)
            self._stream = stream
            self.active_device = sd.query_devices(stream.device)["name"]
            log.info("Speaker open: %s @ %d Hz", self.active_device, rate)
            return
        log.error("SPEAKER UNAVAILABLE; tutor audio will not play")

    def _set_rate(self, rate: int) -> None:
        """Convert audio still queued when the new device plays at a different rate than the old one."""
        with self._lock:
            if rate == self._rate:
                return
            chunks = list(self._chunks)
            if chunks:
                chunks[0] = chunks[0][self._pos:]
            self._chunks = deque(_resample(c, self._rate, rate) for c in chunks)
            self._pos = 0
            self._rate = rate

    # ---------------------------------------------------------------- playback

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - filled, len(chunk) - self._pos)
                out[filled:filled + take] = chunk[self._pos:self._pos + take]
                filled += take
                self._pos += take
                if self._pos >= len(chunk):
                    self._chunks.popleft()
                    self._pos = 0
            if filled and self.first_play_t is None:
                self.first_play_t = time.monotonic()
            if not self._chunks:
                self._drained.set()
        out[filled:] = 0.0
        self._update_level(float(np.sqrt(np.mean(out * out))) if filled else 0.0)

    def _update_level(self, rms: float) -> None:
        target = min(1.0, rms * LEVEL_GAIN)
        if target >= self._level:
            self._level = target
        else:
            self._level = self._level * LEVEL_RELEASE + target * (1.0 - LEVEL_RELEASE)
            if self._level < 0.005:
                self._level = 0.0
        if self._state is not None:
            self._state.tutor_level = self._level

    def enqueue(self, audio: np.ndarray, cancel: threading.Event | None = None) -> bool:
        """Queue 24 kHz audio for playback. Returns False (and drops it) if `cancel` is already set."""
        if len(audio) == 0:
            return False
        if self._stream is None or not self._stream.active:
            log.warning("speaker stream inactive; reopening")
            self.reopen()
        rate = self._rate
        data = _resample(audio, RATE, rate)
        with self._lock:
            if cancel is not None and cancel.is_set():
                return False
            if self._rate != rate:   # the device changed while resampling
                data = _resample(audio, RATE, self._rate)
            self._chunks.append(data.astype(np.float32, copy=False))
            self._drained.clear()
        return True

    def play_test(self) -> bool:
        """Play a short chime so the student can hear which device is in use. Skipped while other audio plays."""
        if not self._drained.is_set():
            return False
        return self.enqueue(_chime())

    def stop(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._pos = 0
            self._drained.set()
        self._level = 0.0
        if self._state is not None:
            self._state.tutor_level = 0.0

    def reset_marker(self) -> None:
        self.first_play_t = None

    def wait_done(self, cancel: threading.Event) -> None:
        while not self._drained.wait(0.02):
            if cancel.is_set():
                return

    def close(self) -> None:
        self.stop()
        self.close_stream()
