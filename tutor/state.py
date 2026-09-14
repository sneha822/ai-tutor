"""
State shared between the conversation loop, the mic, hotkeys, the focus interventions, and the UI.
Plain attributes: each is written by one owner and read by others, which is safe under the GIL.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SharedState:
    turn_active: bool = False              # a reply is being transcribed, generated, or spoken
    tutor_speaking: bool = False           # tutor audio is queued or playing (mic soft gate reads this)
    playback_ended_at: float = 0.0         # monotonic time the tutor last stopped speaking
    user_speaking: bool = False            # the mic is in the middle of an utterance
    interventions_suppressed: bool = False  # automatic focus interventions blocked (hotkey toggle)
    tutor_level: float = 0.0               # 0..1 loudness of the audio playing right now (drives the avatar's mouth)
    user_level: float = 0.0                # 0..1 loudness of the mic (the avatar reacts while the student talks)
    camera_enabled: bool = True            # False = webcam released, focus tracking and auto nudges paused
    mic_enabled: bool = True               # False = mic released, nothing is transcribed
    last_stt_source: str = ""              # "groq" or "local" for the most recent transcription (privacy panel)
