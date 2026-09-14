"""
Conversation loop: utterance -> STT -> tutor (RAG + streaming LLM) -> sentence chunks -> Kokoro -> speaker.

Interruptible at any point: interrupt() silences playback within one audio block, cancels the LLM stream,
and drops queued TTS. A new utterance that completes while a turn is still in progress supersedes it, so
replies are never stale. Focus interventions arrive via intervene() and are spoken like any other turn.
Logs per-stage latency measured from the moment the user stopped speaking.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from tutor import prompts
from tutor.audio.mic import Utterance
from tutor.speech_text import SentenceChunker, to_speech
from tutor.state import SharedState

log = logging.getLogger("voice")

TARGET_MS = 1500


@dataclass
class Intervention:
    message: str          # hidden [SYSTEM: ...] text for the tutor
    queued_t: float


class VoiceLoop:
    def __init__(self, state: SharedState, tutor, stt, tts, speaker,
                 on_event: Callable[[str, str], None] | None = None):
        self._state = state
        self._tutor = tutor
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._on_event = on_event
        self._inbox: queue.Queue[Utterance | Intervention | None] = queue.Queue()
        self._cancel = threading.Event()
        self._carry = ""  # transcript of an utterance that was superseded before it could be answered
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)

    # ------------------------------------------------------------ called from other threads

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.interrupt("shutdown")
        self._inbox.put(None)
        self._thread.join(timeout=3)

    def is_idle(self) -> bool:
        s = self._state
        return not (s.turn_active or s.tutor_speaking or s.user_speaking) and self._inbox.empty()

    def intervene(self, message: str, force: bool = False) -> None:
        """Have the tutor respond to a hidden app message. force=True cuts off anything in progress first."""
        if force:
            self.interrupt("forced intervention")
        self._inbox.put(Intervention(message, time.monotonic()))

    def on_utterance(self, utt: Utterance) -> None:
        if self._state.turn_active:
            # The student finished speaking again while the previous utterance is still being handled
            # (often a pause mid-sentence). Answering the old one afterwards would be stale.
            self.interrupt("new utterance")
        self._inbox.put(utt)

    def on_user_speaking(self) -> None:
        if self._state.turn_active:
            self.interrupt("user started speaking")

    def interrupt(self, source: str = "manual") -> None:
        if not (self._state.turn_active or self._state.tutor_speaking):
            log.info("interrupt via %s: nothing to interrupt", source)
            return
        log.info("INTERRUPT via %s", source)
        self._cancel.set()
        self._speaker.stop()
        self._end_speaking()
        self._emit("interrupted", source)

    # ------------------------------------------------------------------- internal

    def _emit(self, kind: str, text: str) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, text)
        except Exception:
            log.exception("event handler failed (continuing)")

    def _end_speaking(self) -> None:
        if self._state.tutor_speaking:
            self._state.tutor_speaking = False
            self._state.playback_ended_at = time.monotonic()

    def _next_item(self) -> Utterance | Intervention | None:
        item = self._inbox.get()
        if not isinstance(item, Utterance):
            return item
        # Several utterances already waiting means the student kept talking: answer them together.
        while True:
            try:
                nxt = self._inbox.get_nowait()
            except queue.Empty:
                return item
            if not isinstance(nxt, Utterance):
                self._inbox.put(nxt)   # keep the intervention / shutdown sentinel for the next round
                return item
            log.info("merging queued utterance (%.1fs) into the previous one", nxt.duration_s)
            item = Utterance(np.concatenate([item.audio, nxt.audio]), nxt.speech_end_t, nxt.emitted_t,
                             item.duration_s + nxt.duration_s)

    def _run(self) -> None:
        while True:
            item = self._next_item()
            if item is None:
                return
            try:
                if isinstance(item, Intervention):
                    self._intervention_turn(item)
                else:
                    self._turn(item)
            except Exception:
                log.exception("TURN FAILED (continuing)")
            finally:
                self._state.turn_active = False
                self._end_speaking()

    def _turn(self, utt: Utterance) -> None:
        cancel = threading.Event()
        self._cancel = cancel
        self._state.turn_active = True
        marks: dict[str, float] = {}

        text, source, _ = self._stt.transcribe(utt.audio)
        marks["stt_done"] = time.monotonic()
        self._state.last_stt_source = source
        if cancel.is_set():
            if text:
                # Superseded before it was answered: keep it as the start of the student's next utterance.
                self._carry = f"{self._carry} {text}".strip()
            return
        if not text:
            return
        if self._carry:
            text, self._carry = f"{self._carry} {text}", ""
            log.info("student (merged with superseded utterance): %r", text)
        self._emit("user", text)
        self._reply(self._tutor.respond(text, cancel), cancel, marks)
        self._log_latency(utt, source, marks, cancel.is_set())

    def _intervention_turn(self, item: Intervention) -> None:
        if self._state.user_speaking:
            log.info("dropping focus intervention: the student started speaking")
            return
        cancel = threading.Event()
        self._cancel = cancel
        self._state.turn_active = True
        log.info("speaking focus intervention: %s", item.message)
        self._emit("intervention", item.message)
        self._reply(self._tutor.respond(item.message, cancel, hidden=True), cancel, {})
        play = self._speaker.first_play_t
        if play is not None:
            log.info("intervention audio started %.0fms after it was triggered", (play - item.queued_t) * 1000)

    def _reply(self, deltas: Iterator[str], cancel: threading.Event, marks: dict[str, float]) -> None:
        history_before = len(self._tutor.history)
        self._speak_stream(deltas, cancel, marks)
        if cancel.is_set() and len(self._tutor.history) > history_before:
            # Generation often finishes before the student cuts off playback; the tutor must still know.
            self._tutor.mark_last_reply_interrupted()

    def _speak_stream(self, deltas: Iterator[str], cancel: threading.Event, marks: dict[str, float]) -> None:
        chunks: queue.Queue[str | None] = queue.Queue()
        synth = threading.Thread(target=self._synth_worker, args=(chunks, cancel, marks), name="tts", daemon=True)
        self._speaker.reset_marker()
        synth.start()

        def send(chunk: str) -> None:
            chunk = prompts.strip_markers(chunk)
            if not chunk:
                return
            marks.setdefault("chunk_first", time.monotonic())
            chunks.put(chunk)
            self._emit("tutor_chunk", chunk)

        chunker = SentenceChunker()
        reply = []
        # Keep draining after a cancel: the LLM stream stops at its next token, and the tutor then records
        # the reply in history.
        for delta in deltas:
            if cancel.is_set():
                continue
            marks.setdefault("llm_first", time.monotonic())
            reply.append(delta)
            for chunk in chunker.feed(delta):
                send(chunk)
        if not cancel.is_set():
            for chunk in chunker.flush():
                send(chunk)
        chunks.put(None)
        synth.join()
        self._speaker.wait_done(cancel)
        self._end_speaking()
        full = "".join(reply)
        log.info("tutor%s: %r", " (interrupted)" if cancel.is_set() else "", full)
        if not cancel.is_set():
            self._emit("tutor_done", prompts.strip_markers(full))

    def _synth_worker(self, chunks: queue.Queue, cancel: threading.Event, marks: dict[str, float]) -> None:
        while True:
            chunk = chunks.get()
            if chunk is None:
                return
            if cancel.is_set():
                continue
            spoken = to_speech(chunk)
            if not spoken:
                continue
            t0 = time.monotonic()
            try:
                audio = self._tts.synthesize(spoken)
            except Exception:
                log.exception("TTS FAILED for %r (skipping chunk)", spoken)
                continue
            marks.setdefault("tts_first_ms", (time.monotonic() - t0) * 1000)
            if self._speaker.enqueue(audio, cancel):
                marks.setdefault("audio_queued", time.monotonic())
                self._state.tutor_speaking = True

    def _log_latency(self, utt: Utterance, source: str, marks: dict[str, float], cancelled: bool) -> None:
        end = utt.speech_end_t

        def rel(t: float | None) -> str:
            return "-" if t is None else f"{(t - end) * 1000:.0f}"

        play = self._speaker.first_play_t
        log.info("LATENCY ms after user stopped: vad_end=%s stt=%s(%s) llm_first_text=%s first_chunk=%s "
                 "audio_queued=%s audio_playing=%s | first chunk synth=%s%s",
                 rel(utt.emitted_t), rel(marks.get("stt_done")), source, rel(marks.get("llm_first")),
                 rel(marks.get("chunk_first")), rel(marks.get("audio_queued")), rel(play),
                 f"{marks['tts_first_ms']:.0f}" if "tts_first_ms" in marks else "-",
                 " (interrupted)" if cancelled else "")
        if play is not None:
            total = (play - end) * 1000
            log.info("E2E %.0fms %s", total, "OK" if total <= TARGET_MS else f"OVER {TARGET_MS}ms TARGET")
