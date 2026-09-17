"""
Conversation core shared by the text and voice front-ends: session persona + RAG context + recent history +
streaming LLM.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator

import config
from tutor import prompts
from tutor.llm import LLMClient, LLMError
from tutor.session import Session

log = logging.getLogger("tutor")

DEFAULT_TOPIC = "what we were just discussing"
# Messages that probably need several steps (notes, documents, links): these get deeper reasoning.
AGENT_HINT = re.compile(r"\b(notes?|resume|cv|links?|urls?|websites?|site|github|gitlab|portfolio|linkedin|pdf|"
                        r"upload\w*|sections?|pages?|documents?|files?|repos?|repositor\w+|projects?)\b|https?://|www\.",
                        re.I)


class Tutor:
    def __init__(self, context: Callable[[], dict] | None = None, notes=None):
        """context(), if given, returns {"camera_on": bool, "observation": str | None} for the current moment.
        notes, if given, is a tutor.notes.NotesTools: the AI then sees the notes shelf and can open notes and links."""
        self.notes = notes
        self.on_think: Callable[[str, dict], None] | None = None   # thinking events for the page
        self.llm = LLMClient()
        self.retriever = None
        try:
            from tutor.rag import Retriever
            self.retriever = Retriever()
        except Exception:
            log.exception("RAG UNAVAILABLE; continuing without course materials")
        self.history: list[dict] = []
        self.session: Session | None = None
        self.current_topic = DEFAULT_TOPIC   # used in focus interventions; set from the subject or matched sections
        self._context = context
        # Set when the student cut off the last reply. The model is told in the student's next message rather
        # than by editing the reply, because it imitates any marker it sees in its own past replies.
        self._interrupted = False

    def start_session(self, session: Session) -> None:
        """A fresh conversation for a new session (mode, name or subject)."""
        self.session = session
        self.history = []
        self._interrupted = False
        self.current_topic = session.subject or DEFAULT_TOPIC
        if self.notes is not None:
            self.notes.clear()
        log.info("session started: %s", session.label())

    def mark_last_reply_interrupted(self) -> None:
        """The student cut off the last reply (possibly after it finished generating)."""
        self._interrupted = True

    def _camera(self) -> tuple[bool, str | None]:
        if self._context is None:
            return False, None
        try:
            ctx = self._context()
            return bool(ctx.get("camera_on")), ctx.get("observation") or None
        except Exception:
            log.exception("camera context failed (continuing without it)")
            return False, None

    def _thinker(self) -> Callable[[str, dict], None]:
        """Forwards one reply's thinking events to on_think, merging reasoning tokens into small chunks."""
        sink = self.on_think
        buf: list[str] = []
        everything: list[str] = []
        t0 = last = time.monotonic()

        def flush() -> None:
            nonlocal last
            if buf:
                sink("reasoning", {"text": "".join(buf)})
                everything.extend(buf)
                buf.clear()
            last = time.monotonic()

        def think(kind: str, data: dict) -> None:
            if sink is None:
                return
            try:
                if kind == "reasoning":
                    buf.append(data.get("text", ""))
                    if sum(map(len, buf)) >= 48 or time.monotonic() - last > 0.12:
                        flush()
                    return
                flush()
                if kind in ("answer", "end"):
                    data = {**data, "ms": round((time.monotonic() - t0) * 1000)}
                if kind == "end":   # the whole reasoning once more, so a reloaded page can show it
                    data["reasoning"] = "".join(everything)[-4000:]
                sink(kind, data)
            except Exception:
                log.exception("thinking event failed (continuing)")

        return think

    def respond(self, user_text: str, cancel: threading.Event | None = None, hidden: bool = False) -> Iterator[str]:
        """Stream the reply, starting with an emotion tag like [happy].

        hidden=True sends an app message (greeting, focus nudge, check-in) instead of the user's speech: no
        retrieval, no camera note, and the topic is left alone. On any failure, logs loudly and yields a short
        spoken apology instead.
        """
        t0 = time.perf_counter()
        chunks = []
        reply: list[str] = []
        session = self.session or Session()
        camera_on, observation = self._camera()
        if hidden or not self._interrupted:
            content = user_text
        else:
            content = f"{prompts.INTERRUPTED_NOTE} {user_text}"
        if not hidden and camera_on:
            content = prompts.with_camera_note(content, observation)
        think = self._thinker()
        effort = config.LLM_REASONING_EFFORT
        try:
            if not hidden and session.tutoring:
                # Include the previous question so follow-ups like "show me an example of that" still retrieve.
                prev_user = next((m["content"] for m in reversed(self.history)
                                  if m["role"] == "user" and not m["content"].startswith("[SYSTEM:")), "")
                prev_user = prompts.strip_markers(prev_user)
                query = f"{prev_user}\n{user_text}" if prev_user else user_text
                chunks = self.retriever.retrieve(query) if self.retriever else []
                if chunks and chunks[0].section:
                    # Only a matched course section becomes the topic; small talk ("I'm watching the screen") must not.
                    self.current_topic = chunks[0].section[:80].rstrip(" .?!,;:")
            if self.notes is not None and not hidden:
                self.notes.remember_user_text(user_text)
            notes = self.notes if self.notes is not None and self.notes.available else None
            notes_prompt = ""
            tools = None
            if notes is not None:
                from tutor.notes import TOOLS as tools
                notes.begin_turn()
                notes_prompt = prompts.notes_section(notes.library.shelf(), notes.prompt())
                if AGENT_HINT.search(user_text):
                    effort = config.LLM_REASONING_EFFORT_AGENT
            messages = [{"role": "system", "content": prompts.system_prompt(session, chunks, camera_on, notes_prompt)}]
            messages += self.history[-2 * config.HISTORY_TURNS:]
            messages.append({"role": "user", "content": content})
            think("start", {"effort": effort})
            for delta in self.llm.stream(messages, cancel, tools=tools, runner=notes, effort=effort, on_think=think):
                reply.append(delta)
                yield delta
            think("end", {"effort": effort})
        except LLMError:
            think("end", {"effort": effort, "failed": True})
            yield prompts.LLM_FAILURE_REPLY
            return
        except Exception:
            log.exception("TURN FAILED (continuing)")
            think("end", {"effort": effort, "failed": True})
            yield prompts.LLM_FAILURE_REPLY
            return

        if cancel is not None and cancel.is_set():
            self._interrupted = True
        elif not hidden:
            self._interrupted = False   # the note (if any) was delivered with this message
        text = prompts.strip_markers("".join(reply))   # emotion tags stay, so the model keeps using them
        self.history += [{"role": "user", "content": content},
                         {"role": "assistant", "content": text or "[neutral] (no reply)"}]
        log.info("turn done in %.0fms (mode=%s, camera=%s, context chunks=%d, history msgs=%d%s)",
                 (time.perf_counter() - t0) * 1000, session.mode, "on" if camera_on else "off", len(chunks),
                 len(self.history), ", hidden" if hidden else "")
