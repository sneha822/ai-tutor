"""
Microphone capture + Silero VAD -> utterances.

Normal: speech starts when a frame is loud enough (MIN_SPEECH_RMS) and its VAD probability reaches
VAD_THRESHOLD; the utterance ends after END_SILENCE_MS of silence and goes to on_utterance. Once enough
loud speech has accumulated, on_user_speaking fires (the loop uses it to cancel a reply still being generated).

While the tutor is speaking (soft gate): a frame only counts if it is confidently speech AND loud enough.
BARGE_IN_MIN_MS of such frames fires on_barge_in, and the captured audio becomes the start of the new
utterance. With BARGE_IN_ENABLED = False the mic is ignored entirely during playback (hard gate).

The mic can be turned off at runtime (set_enabled): the input device is released and nothing is heard.
It can also be switched to another input (set_device); anything half-heard on the old one is dropped.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch

import config
from tutor.audio.devices import find_device
from tutor.state import SharedState

log = logging.getLogger("mic")

RATE = 16000
FRAME = 512                      # Silero VAD requires 512-sample frames at 16 kHz
FRAME_MS = FRAME * 1000 / RATE   # 32 ms
TRAILING_PAD_FRAMES = 3          # silence kept after speech when handing audio to STT
SILENT_MIC_CHECK_S = 3.0
MIC_LEVEL_GAIN = 8.0             # speech RMS ~0.02-0.12 -> UI level ~0.15-1.0


@dataclass
class Utterance:
    audio: np.ndarray        # float32 mono 16 kHz
    speech_end_t: float      # monotonic time the user stopped speaking
    emitted_t: float         # monotonic time the VAD declared the utterance over
    duration_s: float


class MicListener:
    def __init__(self, state: SharedState, on_utterance: Callable[[Utterance], None],
                 on_user_speaking: Callable[[], None] | None = None,
                 on_barge_in: Callable[[], None] | None = None, device: str | None = config.INPUT_DEVICE):
        self._state = state
        self._on_utterance = on_utterance
        self._on_user_speaking = on_user_speaking
        self._on_barge_in = on_barge_in
        self._q: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=4000)
        self._stop = threading.Event()
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._in_rate = RATE
        self._pending = np.zeros(0, np.float32)
        self._last_status_log = 0.0
        self._vad = None
        self._enabled = True
        self.device = device                # chosen input name (None = system default)
        self.active_device = ""             # name of the device actually open
        self._stream_lock = threading.Lock()
        self._flush = threading.Event()     # set when the input changes: the loop drops anything half-heard

    # ------------------------------------------------------------------ public

    def start(self, open_device: bool = True) -> None:
        from silero_vad import load_silero_vad
        self._vad = load_silero_vad()
        if open_device:
            with self._stream_lock:
                self._stream = self._open()
        self._thread = threading.Thread(target=self._run, args=(open_device,), name="mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.close_stream()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        """Mic off releases the input device and drops anything half-heard; back on starts listening again."""
        if on == self._enabled:
            return
        self._enabled = on
        with self._stream_lock:
            if on:
                try:
                    if self._stream is None:
                        self._stream = self._open()
                    else:
                        self._stream.start()
                except Exception:
                    log.exception("mic restart failed; reopening the device")
                    self._close_locked()
                    self._stream = self._open()
            elif self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    log.exception("mic stop failed (audio is ignored anyway)")
        if on:
            log.info("Mic turned ON")
        else:
            self._state.user_speaking = False
            self._state.user_level = 0.0
            log.info("Mic turned OFF: the tutor is not listening")

    def set_device(self, name: str | None) -> None:
        """Switch to another input (None = system default). While the mic is off, it opens there when turned on."""
        self.device = name
        log.info("Mic switching to %s", name or "system default")
        self.reopen()

    def reopen(self) -> None:
        """Close and reopen the input, after a device switch or a device rescan. Stays closed while the mic is off."""
        with self._stream_lock:
            self._close_locked()
            if self._enabled:
                self._stream = self._open()
        self._flush.set()

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

    def inject(self, audio16: np.ndarray) -> None:
        """Feed 16 kHz audio as if it came from the mic (for tests)."""
        for i in range(0, len(audio16), FRAME):
            self._q.put((audio16[i:i + FRAME].astype(np.float32), RATE))

    # ---------------------------------------------------------------- internal

    def _open(self) -> sd.InputStream | None:
        """Open the chosen input at 16 kHz, or at its native rate (resampled). Caller holds _stream_lock."""
        device = find_device(self.device, "input")
        try:
            native = int(sd.query_devices(device, "input")["default_samplerate"])
        except Exception:
            native = 48000
        for rate in dict.fromkeys((RATE, native)):
            try:
                stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                                        blocksize=int(rate * FRAME / RATE), callback=self._callback)
                stream.start()
                self._in_rate = rate
                self.active_device = sd.query_devices(stream.device)["name"]
                log.info("Mic open: %s @ %d Hz", self.active_device, rate)
                return stream
            except Exception as e:
                log.warning("Mic open at %d Hz failed: %s", rate, e)
        log.error("MIC UNAVAILABLE; voice input disabled")
        return None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and time.monotonic() - self._last_status_log > 5:
            self._last_status_log = time.monotonic()
            log.warning("mic status: %s", status)
        try:
            self._q.put_nowait((indata[:, 0].copy(), self._in_rate))
        except queue.Full:
            pass

    def _frames(self, block: np.ndarray, rate: int):
        if rate != RATE:
            n = int(round(len(block) * RATE / rate))
            block = np.interp(np.linspace(0, len(block) - 1, n), np.arange(len(block)), block)
        self._pending = np.concatenate([self._pending, block.astype(np.float32, copy=False)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            yield frame

    def _drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _fire(self, cb, *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            log.exception("mic callback failed (continuing)")

    def _run(self, check_silent: bool) -> None:
        preroll: deque[np.ndarray] = deque(maxlen=max(1, int(config.PRE_ROLL_MS / FRAME_MS)))
        barge: deque[tuple[np.ndarray, bool]] = deque(maxlen=int(config.BARGE_IN_MIN_MS / FRAME_MS * 1.5) + 1)
        lookback: deque[np.ndarray] = deque(maxlen=max(1, int(config.BARGE_IN_LOOKBACK_MS / FRAME_MS)))
        buf: list[np.ndarray] = []
        in_speech = notified = False
        speech_ms = silence_ms = loud_ms = 0.0
        started = time.monotonic()
        heard_anything = not check_silent
        was_enabled = True

        while not self._stop.is_set():
            if self._flush.is_set() or was_enabled != self._enabled:
                # Mic turned off or on, or switched to another device: forget anything half-heard.
                self._flush.clear()
                was_enabled = self._enabled
                buf, in_speech, notified = [], False, False
                speech_ms = silence_ms = loud_ms = 0.0
                preroll.clear()
                barge.clear()
                lookback.clear()
                self._pending = np.zeros(0, np.float32)
                self._drain()
                self._vad.reset_states()
                self._state.user_speaking = False
                started, heard_anything = time.monotonic(), not check_silent   # re-check the new input for silence
            if not self._enabled:
                try:
                    self._q.get(timeout=0.2)   # discard audio while off
                except queue.Empty:
                    pass
                continue

            try:
                block, rate = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            for frame in self._frames(block, rate):
                try:
                    now = time.monotonic()
                    rms = float(np.sqrt(np.mean(frame ** 2)))
                    if not heard_anything:
                        if rms > 1e-5:
                            heard_anything = True
                        elif now - started > SILENT_MIC_CHECK_S:
                            log.error("MIC IS PERFECTLY SILENT: pick another microphone in the page, or grant "
                                      "microphone permission to your terminal app (macOS: System Settings > Privacy "
                                      "& Security > Microphone) and restart")
                            heard_anything = True
                    with torch.inference_mode():
                        prob = self._vad(torch.from_numpy(frame), RATE).item()
                    st = self._state
                    st.user_speaking = in_speech
                    st.user_level = min(1.0, rms * MIC_LEVEL_GAIN)
                    gated = st.tutor_speaking or now < st.playback_ended_at + config.ECHO_TAIL_MS / 1000

                    if gated and not in_speech:
                        if not config.BARGE_IN_ENABLED:
                            continue
                        qualifies = prob >= config.BARGE_IN_VAD_THRESHOLD and rms >= config.BARGE_IN_MIN_RMS
                        barge.append((frame, qualifies))
                        lookback.append(frame)
                        if sum(q for _, q in barge) * FRAME_MS >= config.BARGE_IN_MIN_MS:
                            log.info("BARGE-IN (prob=%.2f rms=%.3f)", prob, rms)
                            self._fire(self._on_barge_in)
                            # Start the new utterance from the lookback so words before the trigger aren't lost.
                            buf = list(lookback)
                            barge.clear()
                            lookback.clear()
                            in_speech = notified = True
                            speech_ms, silence_ms = config.BARGE_IN_MIN_MS, 0.0
                        continue
                    barge.clear()
                    lookback.clear()

                    if not in_speech:
                        preroll.append(frame)
                        if prob >= config.VAD_THRESHOLD and rms >= config.MIN_SPEECH_RMS:
                            in_speech, notified = True, False
                            buf = list(preroll)
                            speech_ms, silence_ms, loud_ms = FRAME_MS, 0.0, FRAME_MS
                        continue

                    buf.append(frame)
                    if prob >= config.VAD_THRESHOLD - 0.15:   # hysteresis: easier to stay in speech than enter it
                        speech_ms += FRAME_MS
                        silence_ms = 0.0
                        if rms >= config.MIN_SPEECH_RMS:
                            loud_ms += FRAME_MS
                        # Only sustained close-to-mic speech may cancel a reply in progress, not a few loud
                        # syllables in otherwise quiet background talk.
                        if not notified and loud_ms >= config.BARGE_IN_MIN_MS:
                            notified = True
                            self._fire(self._on_user_speaking)
                    else:
                        silence_ms += FRAME_MS

                    too_long = len(buf) * FRAME_MS >= config.MAX_UTTERANCE_S * 1000
                    if silence_ms >= config.END_SILENCE_MS or too_long:
                        in_speech = False
                        loudness = float(np.percentile([np.sqrt(np.mean(f ** 2)) for f in buf], 90))
                        if speech_ms >= config.MIN_UTTERANCE_MS and loudness >= config.MIN_SPEECH_RMS:
                            keep = len(buf) - int(silence_ms / FRAME_MS) + TRAILING_PAD_FRAMES
                            audio = np.concatenate(buf[:max(1, keep)])
                            utt = Utterance(audio, now - silence_ms / 1000, now, len(audio) / RATE)
                            log.info("utterance %.2fs (speech %.0fms, loudness p90 rms=%.3f)",
                                     utt.duration_s, speech_ms, loudness)
                            self._fire(self._on_utterance, utt)
                        elif speech_ms >= config.MIN_UTTERANCE_MS:
                            log.info("ignored quiet utterance (loudness p90 rms=%.3f < MIN_SPEECH_RMS %.3f; "
                                     "likely background speech)", loudness, config.MIN_SPEECH_RMS)
                        buf = []
                        preroll.clear()
                        self._vad.reset_states()
                except Exception:
                    log.exception("mic frame processing failed (continuing)")
