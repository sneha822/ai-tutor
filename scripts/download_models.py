"""
Download every model the app uses, so at the venue only the LLM (and Groq STT, when reachable) needs network.
Run once while online:

    .venv/bin/python scripts/download_models.py
"""
import os
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def face():
    from tutor.focus import ensure_model
    ensure_model()


def embeddings():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(config.EMBED_MODEL, device="cpu")


def spacy_model():
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")


def kokoro():
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cpu")
    list(pipe("Hello.", voice=config.TTS_VOICE))


def whisper():
    from faster_whisper import WhisperModel
    WhisperModel(config.STT_LOCAL_MODEL, device="cpu", compute_type="int8")


def silero():
    from silero_vad import load_silero_vad
    load_silero_vad()


def main():
    results = []
    for name, fn in [("MediaPipe Face Landmarker", face), (f"Embeddings ({config.EMBED_MODEL})", embeddings),
                     ("spaCy en_core_web_sm (Kokoro text front end)", spacy_model),
                     (f"Kokoro-82M + voice {config.TTS_VOICE}", kokoro),
                     (f"faster-whisper {config.STT_LOCAL_MODEL}", whisper), ("Silero VAD", silero)]:
        try:
            fn()
            print(f"[OK]   {name}", flush=True)
            results.append(True)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            results.append(False)
    print("\nAll models cached." if all(results) else "\nSome downloads FAILED; fix before going offline.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
