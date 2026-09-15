"""
Speech-to-text: Groq Whisper (primary) with a local faster-whisper backup.

The backup only starts when Groq hasn't answered within STT_LOCAL_HEDGE_S (or has failed), so normal turns don't
spend CPU on it while the reply's voice is being generated. The local transcript is used when Groq fails, is slower
than STT_GROQ_TIMEOUT_S, or failed recently (STT_OFFLINE_RETRY_S); in that last case the backup runs right away.
Never raises.
"""
from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq

import config
from tutor.perf import WHISPER_THREADS

log = logging.getLogger("stt")

ROOT = Path(__file__).resolve().parent.parent.parent
RATE = 16000
# Whisper's classic outputs on near-silence; ignored for short utterances.
_HALLUCINATIONS = {"", ".", "you", "you.", "thank you", "thank you.", "thanks.", "thanks for watching!",
                   "thanks for watching.", "bye.", "okay.", "so"}


class Transcriber:
    def __init__(self):
        load_dotenv(ROOT / ".env")
        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY") or "missing", max_retries=0,
                          timeout=config.STT_GROQ_TIMEOUT_S + 1.0)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stt")
        self._groq_down_until = 0.0
        self._local = self._load_local()

    def _load_local(self):
        t0 = time.time()
        try:
            from faster_whisper import WhisperModel
            options = {"device": "cpu", "compute_type": "int8", "cpu_threads": WHISPER_THREADS}
            try:
                model = WhisperModel(config.STT_LOCAL_MODEL, local_files_only=True, **options)
            except Exception:
                log.warning("faster-whisper %s not cached; downloading", config.STT_LOCAL_MODEL)
                model = WhisperModel(config.STT_LOCAL_MODEL, **options)
            list(model.transcribe(np.zeros(RATE, np.float32), language="en", beam_size=1)[0])
            log.info("local STT (%s) ready in %.1fs", config.STT_LOCAL_MODEL, time.time() - t0)
            return model
        except Exception:
            log.exception("LOCAL STT UNAVAILABLE; offline speech input will not work")
            return None

    def _groq_stt(self, audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, RATE, format="WAV", subtype="PCM_16")
        r = self._groq.audio.transcriptions.create(model=config.STT_GROQ_MODEL, language="en", temperature=0.0,
                                                   file=("utterance.wav", buf.getvalue()))
        return r.text.strip()

    def _local_stt(self, audio: np.ndarray) -> str:
        segments, _ = self._local.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def _start_local(self, audio: np.ndarray):
        return self._pool.submit(self._local_stt, audio) if self._local is not None else None

    def transcribe(self, audio: np.ndarray) -> tuple[str, str, float]:
        """Return (text, source, milliseconds). Source is groq, local, or none."""
        t0 = time.perf_counter()
        local = None
        text, source = "", "none"

        if time.monotonic() >= self._groq_down_until:
            groq = self._pool.submit(self._groq_stt, audio)
            wait = config.STT_LOCAL_HEDGE_S
            for last_try in (False, True):
                try:
                    text, source = groq.result(timeout=wait), "groq"
                    break
                except FutureTimeout:
                    if last_try:
                        log.error("GROQ STT SLOW (>%.1fs); using local transcript", config.STT_GROQ_TIMEOUT_S)
                        break
                    # Slower than usual: start the backup now, but still take Groq's answer if it lands first.
                    log.info("groq STT not back after %.1fs; starting local backup", wait)
                    local = self._start_local(audio)
                    wait = max(0.05, config.STT_GROQ_TIMEOUT_S - config.STT_LOCAL_HEDGE_S)
                except Exception as e:
                    self._groq_down_until = time.monotonic() + config.STT_OFFLINE_RETRY_S
                    log.error("GROQ STT FAILED (%s: %s); local-only for %ds",
                              type(e).__name__, str(e)[:160], config.STT_OFFLINE_RETRY_S)
                    break

        if source == "none":
            local = local or self._start_local(audio)
            if local is not None:
                try:
                    text, source = local.result(timeout=15), "local"
                except Exception:
                    log.exception("LOCAL STT FAILED")

        duration = len(audio) / RATE
        if text.strip().lower() in _HALLUCINATIONS and duration < 2.0:
            if text:
                log.info("ignoring likely Whisper hallucination %r (%.1fs audio)", text, duration)
            text = ""
        ms = (time.perf_counter() - t0) * 1000
        log.info("stt %.0fms via %s (%.1fs audio): %r", ms, source, duration, text)
        return text, source, ms
