"""
Stage 2 check: terminal chat with the tutor (Groq streaming + RAG over ./materials).

    .venv/bin/python scripts/text_chat.py

Commands: /reindex (pick up changed materials)   /quit
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.tutor import Tutor  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)-6s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    tutor = Tutor()
    print("\nTutor ready. Ask a question. (/reindex, /quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reindex":
            if tutor.retriever:
                tutor.retriever.sync()
            continue
        started = False
        for delta in tutor.respond(text):
            if not started:  # print the prefix only once text arrives, so log lines don't split the reply
                print("tutor> ", end="", flush=True)
                started = True
            print(delta, end="", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
