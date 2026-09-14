"""
Emotion tags. The model starts each reply with a tag like [happy] (and may switch partway through). The tags drive
the 3D face and the page's mood colour; they are never spoken or shown as text.
"""
from __future__ import annotations

import re

EMOTIONS = ("neutral", "happy", "excited", "proud", "playful", "caring", "thoughtful", "confused", "surprised",
            "sad", "annoyed", "angry")
DEFAULT_EMOTION = "neutral"

_TAG = re.compile(r"\[\s*([A-Za-z]{3,12})\s*\]")      # one bracketed word: [happy], [laughs]
_PARTIAL = re.compile(r"\[\s*[A-Za-z]{0,12}\s*$")     # a tag still arriving at the end of the stream


class EmotionTagParser:
    """Splits streamed reply text into ("text", str) and ("emotion", name) pieces, in order.

    A tag split across deltas is held back until it completes. Unknown one-word tags such as [laughs] are dropped,
    so they are never spoken.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self._buf += delta
        out: list[tuple[str, str]] = []
        while (m := _TAG.search(self._buf)) is not None:
            if m.start():
                out.append(("text", self._buf[:m.start()]))
            name = m.group(1).lower()
            if name in EMOTIONS:
                out.append(("emotion", name))
            self._buf = self._buf[m.end():]
        partial = _PARTIAL.search(self._buf)
        cut = partial.start() if partial else len(self._buf)
        if cut:
            out.append(("text", self._buf[:cut]))
            self._buf = self._buf[cut:]
        return out

    def flush(self) -> list[tuple[str, str]]:
        rest, self._buf = self._buf, ""
        return [("text", rest)] if rest else []


def strip_tags(text: str) -> str:
    """Text without emotion tags (or other one-word tags), for terminals, logs and transcripts."""
    return re.sub(r"[ \t]{2,}", " ", _TAG.sub("", text)).strip()


def first_emotion(text: str) -> str | None:
    for m in _TAG.finditer(text):
        name = m.group(1).lower()
        if name in EMOTIONS:
            return name
    return None
