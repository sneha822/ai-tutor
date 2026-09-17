"""
CPU thread budgets. Embeddings (torch), faster-whisper (CTranslate2), MediaPipe and the browser's 3D page all want the
CPU at the same moment, right when a reply is being voiced, and each library defaults to about one thread per core.
On laptops, Windows ones especially, that oversubscription is what makes the voice start late. Capping each pool
keeps them from fighting.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("perf")

CORES = os.cpu_count() or 4
TORCH_THREADS = max(2, min(6, CORES // 2))     # embeddings and other torch work
WHISPER_THREADS = max(2, min(4, CORES // 2))   # offline speech recognition backup


def limit_threads() -> None:
    """Call once at startup, before any model loads. Never raises."""
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass   # only settable before torch runs anything in parallel
        log.info("CPU: %d cores; voice uses %d threads, offline STT %d", CORES, TORCH_THREADS, WHISPER_THREADS)
    except Exception:
        log.exception("could not limit CPU threads (continuing)")
