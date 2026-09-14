"""
Conversation core shared by the text and voice front-ends: RAG context + recent history + streaming LLM.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

import config
from tutor import prompts
from tutor.llm import LLMClient, LLMError

log = logging.getLogger("tutor")


class Tutor:
    def __init__(self):
        self.llm = LLMClient()
        self.retriever = None
        try:
            from tutor.rag import Retriever
            self.retriever = Retriever()
        except Exception:
            log.exception("RAG UNAVAILABLE; continuing without course materials")
        self.history: list[dict] = []
        self.current_topic = "what we were just discussing"   # used in focus interventions; set from matched course sections
        # Set when the student cut off the last reply. The model is told in the student's next message rather
        # than by editing the reply, because it imitates any marker it sees in its own past replies.
        self._interrupted = False

    def mark_last_reply_interrupted(self) -> None:
        """The student cut off the last reply (possibly after it finished generating)."""
        self._interrupted = True

    def respond(self, user_text: str, cancel: threading.Event | None = None, hidden: bool = False) -> Iterator[str]:
        """Stream the tutor's reply.

        hidden=True sends an app message (a focus intervention) instead of student speech: no retrieval, and the
        topic is left alone. On any failure, logs loudly and yields a short spoken apology instead.
        """
        t0 = time.perf_counter()
        chunks = []
        reply: list[str] = []
        if hidden or not self._interrupted:
            content = user_text
        else:
            content = f"{prompts.INTERRUPTED_NOTE} {user_text}"
        try:
            if not hidden:
                # Include the previous question so follow-ups like "show me an example of that" still retrieve.
                prev_user = next((m["content"] for m in reversed(self.history)
                                  if m["role"] == "user" and not m["content"].startswith("[SYSTEM:")), "")
                prev_user = prompts.strip_markers(prev_user)
                query = f"{prev_user}\n{user_text}" if prev_user else user_text
                chunks = self.retriever.retrieve(query) if self.retriever else []
                if chunks and chunks[0].section:
                    # Only a matched course section becomes the topic; small talk ("I'm watching the screen") must not.
                    self.current_topic = chunks[0].section[:80].rstrip(" .?!,;:")
            messages = [{"role": "system", "content": prompts.system_prompt(chunks)}]
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
        text = prompts.strip_markers("".join(reply))
        self.history += [{"role": "user", "content": content},
                         {"role": "assistant", "content": text or "(no reply)"}]
        log.info("turn done in %.0fms (context chunks=%d, history msgs=%d%s)",
                 (time.perf_counter() - t0) * 1000, len(chunks), len(self.history), ", hidden" if hidden else "")
