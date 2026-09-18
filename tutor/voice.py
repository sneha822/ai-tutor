"""
Conversation loop: utterance -> STT -> tutor (RAG + streaming LLM) -> sentence chunks -> Kokoro -> speaker.

Interruptible at any point: interrupt() silences playback within one audio block, cancels the LLM stream,
and drops queued TTS. A new utterance that completes while a turn is still in progress supersedes it, so
replies are never stale. Focus interventions arrive via intervene() and are spoken like any other turn.

Nothing is answered until a session starts (the page's welcome form); begin_session() starts a fresh conversation
with a spoken greeting. Emotion tags in replies ([happy]) are stripped from speech and text and shown on the face
when the voice reaches them. Logs per-stage latency measured from the moment the user stopped speaking.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from tutor import prompts
from tutor.audio.mic import Utterance
from tutor.emotions import DEFAULT_EMOTION, EmotionTagParser, strip_tags
from tutor.session import Session
from tutor.speech_text import SentenceChunker, to_speech
from tutor.state import SharedState

log = logging.getLogger("voice")

TARGET_MS = 1500


@dataclass
class Intervention:
    message: str               # hidden [SYSTEM: ...] text for the tutor
    queued_t: float
    label: str | None = None   # what the page shows ("Focus nudge · ..."); None shows nothing
    kind: str = "nudge"        # "nudge" (focus) or "action" (something the user asked for, like Explain)


@dataclass
class SessionStart:
    session: Session
    greeting: str              # hidden [SYSTEM: ...] text that makes the AI open the session


@dataclass
class TextTurn:
    text: str                  # a message typed in the page (or a suggested question clicked)


@dataclass
class EmotionMark:
    name: str                  # queued between speech chunks, so the face changes when the voice gets there


class VoiceLoop:
    def __init__(self, state: SharedState, tutor, stt, tts, speaker,
                 on_event: Callable[[str, str], None] | None = None):
        self._state = state
        self._tutor = tutor
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._on_event = on_event
        self._inbox: queue.Queue[Utterance | Intervention | SessionStart | TextTurn | None] = queue.Queue()
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

    def begin_session(self, session: Session, greeting: str) -> None:
        """Start a fresh conversation: cut off anything in progress, drop stale turns, then greet."""
        self.interrupt("new session")
        self._drop_pending()
        self._state.session = session
        self._inbox.put(SessionStart(session, greeting))

    def intervene(self, message: str, force: bool = False, label: str | None = None, kind: str = "nudge") -> None:
        """Have the tutor respond to a hidden app message. force=True cuts off anything in progress first."""
        if force:
            self.interrupt("forced intervention" if kind == "nudge" else "user action")
        self._inbox.put(Intervention(message, time.monotonic(), label, kind))

    def ask(self, text: str) -> bool:
        """A typed message: answered like speech, cutting off a reply in progress."""
        if self._state.session is None:
            log.info("ignoring typed message: no session yet")
            return False
        self.interrupt("typed message")
        self._inbox.put(TextTurn(text))
        return True

    def on_utterance(self, utt: Utterance) -> None:
        if self._state.session is None:
            log.info("ignoring speech: no session yet (fill in the welcome form in the page)")
            return
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

    def _show_emotion(self, name: str) -> None:
        self._state.emotion = name
        self._state.emotion_changed_at = time.monotonic()
        self._emit("emotion", name)

    def _end_speaking(self) -> None:
        if self._state.tutor_speaking:
            self._state.tutor_speaking = False
            self._state.playback_ended_at = time.monotonic()

    def _drop_pending(self) -> None:
        """Forget queued utterances and interventions (keeping a shutdown request)."""
        shutdown = False
        while True:
            try:
                shutdown = self._inbox.get_nowait() is None or shutdown
            except queue.Empty:
                break
        if shutdown:
            self._inbox.put(None)
        self._carry = ""

    def _next_item(self) -> Utterance | Intervention | SessionStart | None:
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
                self._inbox.put(nxt)   # keep the intervention / session / shutdown sentinel for the next round
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
                if isinstance(item, SessionStart):
                    self._session_turn(item)
                elif isinstance(item, TextTurn):
                    self._text_turn(item)
                elif isinstance(item, Intervention):
                    self._intervention_turn(item)
                else:
                    self._turn(item)
            except Exception:
                log.exception("TURN FAILED (continuing)")
            finally:
                self._state.turn_active = False
                self._end_speaking()

    def _begin_turn(self) -> threading.Event:
        cancel = threading.Event()
        self._cancel = cancel
        self._state.turn_active = True
        return cancel

    def _turn(self, utt: Utterance) -> None:
        cancel = self._begin_turn()
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

    def _text_turn(self, item: TextTurn) -> None:
        cancel = self._begin_turn()
        log.info("student (typed): %r", item.text)
        self._emit("user", item.text)
        self._reply(self._tutor.respond(item.text, cancel), cancel, {})

    def _session_turn(self, item: SessionStart) -> None:
        self._tutor.start_session(item.session)
        cancel = self._begin_turn()
        log.info("greeting for new session: %s", item.session.label())
        self._reply(self._tutor.respond(item.greeting, cancel, hidden=True), cancel, {})

    def _intervention_turn(self, item: Intervention) -> None:
        if self._state.user_speaking and item.kind == "nudge":
            log.info("dropping focus intervention: the student started speaking")
            return
        cancel = self._begin_turn()
        log.info("speaking %s: %s", "focus intervention" if item.kind == "nudge" else "user action", item.message)
        if item.label:
            self._emit("intervention" if item.kind == "nudge" else "action", item.label)
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
        chunks: queue.Queue[str | EmotionMark | None] = queue.Queue()
        synth = threading.Thread(target=self._synth_worker, args=(chunks, cancel, marks), name="tts", daemon=True)
        self._speaker.reset_marker()
        synth.start()
        chunker = SentenceChunker()
        tags = EmotionTagParser()
        felt = sent = False

        def send(chunk: str) -> None:
            nonlocal sent
            chunk = prompts.strip_markers(chunk).replace("**", "")   # the model sometimes bolds words anyway
            if not chunk:
                return
            sent = True
            marks.setdefault("chunk_first", time.monotonic())
            chunks.put(chunk)
            self._emit("tutor_chunk", chunk)

        def feel(name: str) -> None:
            nonlocal felt
            felt = True
            for chunk in chunker.flush():   # a tag starts a new sentence
                send(chunk)
            if sent:
                chunks.put(EmotionMark(name))   # mid-reply: change the face when the voice gets there
            else:
                self._show_emotion(name)        # opening tag: show it before the words appear

        def take(pieces: list[tuple[str, str]]) -> None:
            for kind, value in pieces:
                if kind == "emotion":
                    feel(value)
                    continue
                if not felt and value.strip():
                    feel(DEFAULT_EMOTION)   # the model forgot its tag
                for chunk in chunker.feed(value):
                    send(chunk)

        reply = []
        # Keep draining after a cancel: the LLM stream stops at its next token, and the tutor then records
        # the reply in history.
        for delta in deltas:
            if cancel.is_set():
                continue
            marks.setdefault("llm_first", time.monotonic())
            reply.append(delta)
            take(tags.feed(delta))
        if not cancel.is_set():
            take(tags.flush())
            for chunk in chunker.flush():
                send(chunk)
        chunks.put(None)
        synth.join()
        self._speaker.wait_done(cancel)
        self._end_speaking()
        full = "".join(reply)
        log.info("tutor%s: %r", " (interrupted)" if cancel.is_set() else "", full)
        if not cancel.is_set():
            self._emit("tutor_done", strip_tags(prompts.strip_markers(full)))

    def _synth_worker(self, chunks: queue.Queue, cancel: threading.Event, marks: dict[str, float]) -> None:
        """Starts speech for each chunk as soon as it arrives (several at once with a cloud voice) and hands the
        results to the player in their original order, emotion marks included."""
        ordered: queue.Queue = queue.Queue()
        player = threading.Thread(target=self._play_worker, args=(ordered, cancel, marks), name="tts-play", daemon=True)
        player.start()
        pool = ThreadPoolExecutor(max_workers=max(1, getattr(self._tts, "parallel", 1)), thread_name_prefix="tts")
        first = True
        try:
            while True:
                chunk = chunks.get()
                if chunk is None:
                    break
                if cancel.is_set():
                    continue
                if isinstance(chunk, EmotionMark):
                    ordered.put(chunk)
                    continue
                spoken = to_speech(chunk)
                if spoken:
                    ordered.put(pool.submit(self._synthesize, spoken, marks, first))
                    first = False
        finally:
            ordered.put(None)
            player.join()
            pool.shutdown(wait=False, cancel_futures=True)

    def _synthesize(self, spoken: str, marks: dict[str, float], first: bool) -> np.ndarray | None:
        t0 = time.monotonic()
        try:
            audio = self._tts.synthesize(spoken)
        except Exception:
            log.exception("TTS FAILED for %r (skipping chunk)", spoken)
            return None
        if first:
            marks["tts_first_ms"] = (time.monotonic() - t0) * 1000
        return audio

    def _play_worker(self, ordered: queue.Queue, cancel: threading.Event, marks: dict[str, float]) -> None:
        while True:
            item = ordered.get()
            if item is None:
                return
            if cancel.is_set():
                continue
            if isinstance(item, EmotionMark):
                self._schedule_emotion(item.name, cancel)
                continue
            audio = item.result()
            if audio is not None and len(audio) and self._speaker.enqueue(audio, cancel):
                marks.setdefault("audio_queued", time.monotonic())
                self._state.tutor_speaking = True

    def _schedule_emotion(self, name: str, cancel: threading.Event) -> None:
        """Show an emotion once the audio queued ahead of it has played, so the face changes with the voice."""
        lead = self._speaker.pending_seconds()
        if lead < 0.05:
            self._show_emotion(name)
            return

        def show() -> None:
            if not cancel.is_set():
                self._show_emotion(name)

        timer = threading.Timer(lead, show)
        timer.daemon = True
        timer.start()

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
