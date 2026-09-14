"""
The session chosen in the page's welcome form: who the user is and what the conversation is for.

    tutor      teaches a subject; watches focus through the camera and nudges (sternly, if it keeps happening)
    friend     casual friend with real reactions
    therapist  calm, supportive listener (an AI, not a real therapist)
    chat       friendly conversation about anything
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

MODES = ("tutor", "friend", "therapist", "chat")
PERSONA = {"tutor": "Tutor", "friend": "Friend", "therapist": "Therapist", "chat": "Companion"}
_UNSAFE = re.compile(r"[^\w .,'&+\-/()]")


def _clean(value, limit: int) -> str:
    """Plain name/subject characters only: brackets or newlines could smuggle instructions into the prompt."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", _UNSAFE.sub("", value)).strip()[:limit]


@dataclass(frozen=True)
class Session:
    mode: str = "tutor"
    name: str = ""
    subject: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def persona(self) -> str:
        return PERSONA[self.mode]

    @property
    def tutoring(self) -> bool:
        return self.mode == "tutor"

    def label(self) -> str:
        what = f"tutor ({self.subject or 'any subject'})" if self.tutoring else self.mode
        return f"{what} for {self.name or 'an unnamed user'}"

    def to_dict(self) -> dict:
        return {**asdict(self), "persona": self.persona}

    @classmethod
    def from_request(cls, msg: dict) -> Session | None:
        """Session from the welcome form's message, or None if the mode isn't one we know."""
        mode = msg.get("mode")
        if mode not in MODES:
            return None
        subject = _clean(msg.get("subject"), 60) if mode == "tutor" else ""
        return cls(mode=mode, name=_clean(msg.get("name"), 40), subject=subject)
