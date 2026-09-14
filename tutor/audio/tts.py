"""Kokoro-82M text-to-speech on CPU. Measured ~0.1x real time (a 4.75 s sentence renders in ~460 ms)."""
from __future__ import annotations

import logging
import time

import numpy as np

import config

log = logging.getLogger("tts")

RATE = 24000


class TTS:
    def __init__(self):
        from kokoro import KPipeline
        t0 = time.time()
        self._pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
        self.synthesize("Ready.")
        log.info("TTS ready in %.1fs (voice %s)", time.time() - t0, config.TTS_VOICE)

    def synthesize(self, text: str) -> np.ndarray:
        """float32 mono audio at 24 kHz. Call from one thread at a time."""
        parts = [r.audio.detach().cpu().numpy()
                 for r in self._pipe(text, voice=config.TTS_VOICE, speed=config.TTS_SPEED) if r.audio is not None]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
