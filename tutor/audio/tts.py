"""
The tutor's voice.

TTS_ENGINE = "groq": Groq's Orpheus voice turns each sentence into speech. It's fast on any laptop, and several
sentences can be requested at once (TTS_PARALLEL). If Groq fails, Kokoro-82M on this computer takes over; it is
loaded only when first needed (TTS_LOCAL_PRELOAD = False), because it takes ~1.2 GB of RAM and a while to load.
While it loads, sentences are shown but not spoken.

TTS_ENGINE = "local": Kokoro only (fully offline; needs a fast CPU, ~0.1x real time on Apple Silicon).
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

import config
from tutor import local_settings

log = logging.getLogger("tts")

ROOT = Path(__file__).resolve().parent.parent.parent
RATE = 24000
# Each Orpheus model has its own one-time terms acceptance, so the link follows TTS_GROQ_MODEL.
TERMS_URL = "https://console.groq.com/playground?model=" + config.TTS_GROQ_MODEL.replace("/", "%2F")
TERMS_RETRY_S = 600


_RETRY_IN = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)


def _retry_after(message: str) -> float:
    """Seconds to wait, from Groq's "Please try again in 4m48s" (0 if it didn't say). Capped at an hour."""
    m = _RETRY_IN.search(message)
    if not m or not any(m.groups()):
        return 0.0
    h, mins, secs = (float(g or 0) for g in m.groups())
    return min(h * 3600 + mins * 60 + secs + 1, 3600)


def _friendly(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f}s"


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(audio) == 0:
        return audio
    n = max(1, int(round(len(audio) * dst / src)))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def engine() -> str:
    """"groq" or "local": this computer's choice ("tts_engine" in local_settings.json), else config.TTS_ENGINE."""
    choice = local_settings.load().get("tts_engine")
    return choice if choice in ("groq", "local") else config.TTS_ENGINE


def split_for_voice(text: str, limit: int) -> list[str]:
    """Pieces of at most `limit` characters, cut at sentence or word boundaries."""
    parts, rest = [], text.strip()
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("; "), window.rfind(", "))
        if cut < limit // 3:
            cut = window.rfind(" ")
        cut = cut + 1 if cut > 0 else limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    return parts + ([rest] if rest else [])


class TTS:
    def __init__(self):
        self.engine = engine()
        log.info("voice engine: %s", self.engine)
        self.parallel = config.TTS_PARALLEL if self.engine == "groq" else 1
        self.source = ""                        # "groq" or "local", for the last sentence spoken
        self._local = None
        self._local_lock = threading.Lock()     # Kokoro isn't safe to call from two threads at once
        self._local_loading = threading.Event()
        self._down_until = 0.0
        self._groq = None
        if self.engine == "groq":
            load_dotenv(ROOT / ".env")
            from groq import Groq
            self._groq = Groq(api_key=os.getenv("GROQ_API_KEY") or "missing", max_retries=0,
                              timeout=config.TTS_GROQ_TIMEOUT_S)
            threading.Thread(target=self._warm_up, name="tts-warmup", daemon=True).start()
        if self.engine != "groq" or config.TTS_LOCAL_PRELOAD:
            self._load_local()

    # ------------------------------------------------------------------ public

    def synthesize(self, text: str) -> np.ndarray:
        """float32 mono audio at 24 kHz. Safe to call from several threads. Empty if no voice is available yet."""
        if self._groq is not None and time.monotonic() >= self._down_until:
            try:
                audio = self._groq_tts(text)
                self.source = "groq"
                return audio
            except Exception as e:
                self._groq_failed(e)
        return self._local_tts(text)

    # ------------------------------------------------------------------ Groq

    def _warm_up(self) -> None:
        t0 = time.perf_counter()
        try:
            self._groq_tts("Ready.")
            log.info("Groq voice ready in %.0fms (%s, voice %s)", (time.perf_counter() - t0) * 1000,
                     config.TTS_GROQ_MODEL, config.TTS_GROQ_VOICE)
        except Exception as e:
            self._groq_failed(e)

    def _groq_tts(self, text: str) -> np.ndarray:
        pieces = []
        for part in split_for_voice(text, config.TTS_GROQ_MAX_CHARS):
            response = self._groq.audio.speech.create(model=config.TTS_GROQ_MODEL, voice=config.TTS_GROQ_VOICE,
                                                      input=part, response_format="wav")
            audio, rate = sf.read(io.BytesIO(response.read()), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            pieces.append(_resample(audio, rate, RATE))
        return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)

    def _groq_failed(self, e: Exception) -> None:
        message = str(e)
        if "terms" in message.lower():
            wait = TERMS_RETRY_S
            log.error("GROQ VOICE NEEDS ONE-TIME TERMS ACCEPTANCE: open %s (as the Groq org owner) and accept. "
                      "Using the voice on this computer meanwhile.", TERMS_URL)
        elif "429" in message or "rate" in message.lower():
            # Groq says when the quota frees up ("try again in 4m48s"); the free tier's daily cap is ~3600 tokens
            # per voice model, about an hour of speech.
            wait = _retry_after(message) or config.TTS_OFFLINE_RETRY_S
            log.error("GROQ VOICE RATE LIMITED (%s); using the voice on this computer for %s",
                      "daily quota used up" if "per day" in message or "TPD" in message else "too many requests",
                      _friendly(wait))
        else:
            wait = config.TTS_OFFLINE_RETRY_S
            log.error("GROQ VOICE FAILED (%s: %s); using the voice on this computer for %ds",
                      type(e).__name__, message[:160], wait)
        if time.monotonic() >= self._down_until:
            self._down_until = time.monotonic() + wait
        self._start_local_load()

    # ------------------------------------------------------------------ Kokoro (local)

    def _start_local_load(self) -> None:
        if self._local is None and not self._local_loading.is_set():
            self._local_loading.set()
            threading.Thread(target=self._load_local, name="tts-local", daemon=True).start()

    def _load_local(self) -> None:
        self._local_loading.set()
        t0 = time.time()
        try:
            from kokoro import KPipeline
            pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
            with self._local_lock:
                self._local = pipe
            self._kokoro("Ready.")
            log.info("local voice (Kokoro) ready in %.1fs (voice %s)", time.time() - t0, config.TTS_VOICE)
        except Exception:
            log.exception("LOCAL VOICE UNAVAILABLE")
            self._local_loading.clear()

    def _kokoro(self, text: str) -> np.ndarray:
        with self._local_lock:
            parts = [r.audio.detach().cpu().numpy()
                     for r in self._local(text, voice=config.TTS_VOICE, speed=config.TTS_SPEED) if r.audio is not None]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)

    def _local_tts(self, text: str) -> np.ndarray:
        if self._local is None:
            self._start_local_load()
            log.warning("no voice yet for %r (the local voice is still loading)", re.sub(r"\s+", " ", text)[:40])
            return np.zeros(0, np.float32)
        self.source = "local"
        return self._kokoro(text)
