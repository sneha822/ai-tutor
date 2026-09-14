"""
Conversation core shared by the text and voice front-ends: session persona + RAG context + recent history +
streaming LLM.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator

import config
from tutor import prompts
from tutor.llm import LLMClient, LLMError
from tutor.session import Session

log = logging.getLogger("tutor")

DEFAULT_TOPIC = "what we were just discussing"


class Tutor:
    def __init__(self, context: Callable[[], dict] | None = None):
        """context(), if given, returns {"camera_on": bool, "observation": str | None} for the current moment."""
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
            messages = [{"role": "system", "content": prompts.system_prompt(session, chunks, camera_on)}]
            messages += self.history[-2 * config.HISTORY_TURNS:]
            messages.append({"role": "user", "content": content})
            for delta in self.llm.stream(messages, cancel):
                reply.append(delta)
                yield delta
        except LLMError:
            yield prompts.LLM_FAILURE_REPLY
            return
        except Exception:
            log.exception("TURN FAILED (continuing)")
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
