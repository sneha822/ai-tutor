"""
Stage 2 check: terminal chat (Groq streaming + RAG over ./materials), in any session mode. Emotion tags are shown
as (happy) before the text they belong to.

    .venv/bin/python scripts/text_chat.py --mode tutor --name Sam --subject Calculus
    .venv/bin/python scripts/text_chat.py --mode friend --name Sam

Commands: /add <file> (add a PDF or image to your notes)   /notes   /see <words> (pretend the camera sees e.g.
"smiling, looking at you")   /nocam   /reindex   /quit
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor import prompts  # noqa: E402
from tutor.emotions import EmotionTagParser  # noqa: E402
from tutor.notes import NoteError, NotesLibrary, NotesTools  # noqa: E402
from tutor.session import MODES, Session  # noqa: E402
from tutor.tutor import Tutor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Chat with the AI in the terminal.")
    parser.add_argument("--mode", choices=MODES, default="tutor")
    parser.add_argument("--name", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--no-greeting", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)-6s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    camera = {"camera_on": False, "observation": None}
    tutor = Tutor(context=lambda: camera)
    tutor.notes = NotesTools(NotesLibrary(tutor.retriever))
    session = Session.from_request({"mode": args.mode, "name": args.name, "subject": args.subject})
    tutor.start_session(session)

    def say(text: str, hidden: bool = False) -> None:
        tags = EmotionTagParser()
        started = False
        for delta in tutor.respond(text, hidden=hidden):
            for kind, value in tags.feed(delta):
                if not started:  # print the prefix only once text arrives, so log lines don't split the reply
                    print(f"{session.persona.lower()}> ", end="", flush=True)
                    started = True
                print(f"({value}) " if kind == "emotion" else value, end="", flush=True)
        for _, value in tags.flush():
            print(value, end="")
        print(flush=True)

    print(f"\n{session.persona} ready ({session.label()}). /see <words>, /nocam, /reindex, /quit")
    if not args.no_greeting:
        say(prompts.greeting_message(session), hidden=True)
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
        if text.startswith("/add "):
            path = Path(text[5:].strip()).expanduser()
            try:
                note = tutor.notes.library.add(path.name, path.read_bytes())
                print(f"(added note {note.id}; it's being read in the background, see /notes)")
            except (OSError, NoteError) as e:
                print(f"(couldn't add it: {e})")
            continue
        if text == "/notes":
            print(tutor.notes.library.shelf() or "(no notes yet)")
            continue
        if text.startswith("/see "):
            camera.update(camera_on=True, observation=text[5:].strip())
            print(f"(camera on: {camera['observation']})")
            continue
        if text == "/nocam":
            camera.update(camera_on=False, observation=None)
            print("(camera off)")
            continue
        say(text)


if __name__ == "__main__":
    main()
