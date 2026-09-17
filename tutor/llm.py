"""
Chat client: streaming, cancellable, with latency logging, over the providers in tutor/providers.py (NVIDIA NIM
first, then Groq). Errors are logged loudly and re-raised as LLMError; nothing here crashes the app.

A provider that fails when a request starts (rate limit, outage, unsupported option) is skipped for
LLM_PROVIDER_RETRY_S and the next one takes the request, so a reply still arrives.

Agent mode: with tools, the model's tool calls run through a runner (announce/describe/run) and the conversation
continues in the same stream, for up to LLM_MAX_TOOL_ROUNDS rounds. The runner's short "let me look" line is yielded
while a tool runs, so there is no silent pause. on_think receives the model's reasoning as it streams, a "step"
event before and after each tool, and "answer" when the real reply starts.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator

import config
from tutor import providers

log = logging.getLogger("llm")

_EFFORT_ORDER = ("low", "medium", "high")
# "I'll open your note", "let me check", "one sec" - said without actually calling a tool.
_PROMISE = re.compile(r"\b(?:i'?ll|i will|let me|i'?m going to|i am going to|going to)\b[^.!?\n]{0,40}?"
                      r"\b(?:open|read|check|look|pull|fetch|find|search|dig|review|start with|grab|walk through|"
                      r"go through|take a look|explain)\b"
                      r"|\b(?:hold on|one (?:sec|second|moment)|give me a (?:sec|second|moment)|bear with me)\b", re.I)
_CARRY_ON = "[SYSTEM: You said you were about to do something but didn't do it. Do it now, in this same reply{how}. Don't announce it again.]"
_CARRY_ON_TOOLS = ": call the tools you need, then give the full answer"
MAX_CARRY_ONS = 2


def _only_promised(text: str) -> bool:
    """True when a reply just announces an action (so the model stopped before doing the work)."""
    text = text.strip()
    return bool(text) and len(text) < 400 and _PROMISE.search(text) is not None


class LLMError(Exception):
    pass


class _Retry(Exception):
    """A round stalled before producing anything; it is repeated on the next provider."""


def _deeper(a: str, b: str) -> str:
    return max(a, b, key=lambda e: _EFFORT_ORDER.index(e) if e in _EFFORT_ORDER else 0)


class LLMClient:
    def __init__(self):
        self.routes = providers.routes("chat")
        self._skip_until: dict[str, float] = {}
        if not self.routes:
            log.error("NO AI PROVIDER: add NVIDIA_API_KEY or GROQ_API_KEY to .env; every reply will fail")
        else:
            log.info("AI replies: %s", " -> ".join(r.name for r in self.routes))
        self.warmup()

    def warmup(self) -> None:
        """Open each provider's connection before the first real turn. Never raises."""
        for provider in dict.fromkeys(r.provider for r in self.routes):
            t0 = time.perf_counter()
            try:
                providers.client(provider).models.list()
                log.info("llm connection to %s warm in %.0fms", provider, (time.perf_counter() - t0) * 1000)
            except Exception as e:
                log.error("LLM WARMUP FAILED for %s (%s: %s); first turn may be slow", provider, type(e).__name__, e)

    def _create(self, route: providers.Route, messages: list[dict], tools: list[dict] | None, effort: str):
        options = providers.reasoning_options(route, effort)
        if tools:
            options["tools"] = tools
        return route.client.chat.completions.create(
            model=route.model,
            messages=messages,
            stream=True,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS if effort == "low" else config.LLM_MAX_TOKENS_AGENT,
            **options,
        )

    def _first_route(self) -> int:
        now = time.monotonic()
        for i, route in enumerate(self.routes):
            if self._skip_until.get(route.name, 0) <= now:
                return i
        return len(self.routes) - 1   # everything failed recently: try the last one anyway

    def _fail_over(self, pos: list[int], e: Exception) -> None:
        """Skip the current provider for a while and move to the next; re-raise if there is none."""
        route = self.routes[pos[0]]
        if pos[0] + 1 >= len(self.routes):
            raise e
        self._skip_until[route.name] = time.monotonic() + config.LLM_PROVIDER_RETRY_S
        log.warning("LLM %s FAILED (%s: %s); using %s for %ds", route.name,
                    providers.status(e) or type(e).__name__, str(e)[:140],
                    self.routes[pos[0] + 1].name, config.LLM_PROVIDER_RETRY_S)
        pos[0] += 1

    def _guard(self, resp, said: list[str], calls: dict, pos: list[int]):
        """Pass chunks through; if the stream breaks before anything was said or called, fail over instead."""
        try:
            yield from resp
        except Exception as e:
            if said or calls:
                raise
            self._fail_over(pos, e)
            raise _Retry() from e

    def _open(self, pos: list[int], messages: list[dict], tools: list[dict] | None, effort: str):
        """Start a streaming request, moving down the provider list when one fails to start."""
        while True:
            route = self.routes[pos[0]]
            try:
                return route, self._create(route, messages, tools, effort)
            except Exception as e:
                self._fail_over(pos, e)

    def stream(self, messages: list[dict], cancel: threading.Event | None = None,
               tools: list[dict] | None = None, runner=None, effort: str | None = None,
               on_think: Callable[[str, dict], None] | None = None) -> Iterator[str]:
        """Yield text deltas. Stops early (closing the HTTP stream) once `cancel` is set.

        runner (with tools): announce(name, args) -> short line or None, describe(name, args, done) -> (label, ok),
        run(name, args) -> result text.
        """
        if not self.routes:
            raise LLMError("no AI provider configured")
        t0 = time.perf_counter()
        first_ms = None
        chars = 0
        effort = effort or config.LLM_REASONING_EFFORT
        messages = list(messages)
        used_tools: list[str] = []
        carry_ons = 0
        answered = False
        pos = [self._first_route()]
        route = self.routes[pos[0]]

        def think(kind: str, data: dict) -> None:
            if on_think is not None:
                on_think(kind, data)

        try:
            round_no = 0
            while round_no <= config.LLM_MAX_TOOL_ROUNDS:
                round_no += 1
                offered = tools if tools and runner is not None and round_no <= config.LLM_MAX_TOOL_ROUNDS else None
                round_effort = effort if round_no == 1 else _deeper(effort, config.LLM_REASONING_EFFORT_AGENT)
                route, resp = self._open(pos, messages, offered, round_effort)
                said: list[str] = []
                calls: dict[int, dict] = {}
                try:
                    for chunk in self._guard(resp, said, calls, pos):
                        if cancel is not None and cancel.is_set():
                            log.info("llm cancelled after %.0fms (%d chars)", (time.perf_counter() - t0) * 1000, chars)
                            return
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        reasoning = providers.reasoning_text(choice.delta)
                        if reasoning:
                            think("reasoning", {"text": reasoning})
                        delta = choice.delta.content or ""
                        if delta:
                            if first_ms is None:
                                first_ms = (time.perf_counter() - t0) * 1000
                            if not answered:
                                answered = True
                                think("answer", {})
                            chars += len(delta)
                            said.append(delta)
                            yield delta
                        for tc in choice.delta.tool_calls or []:
                            call = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            call["id"] += tc.id or ""
                            if tc.function:
                                call["name"] += tc.function.name or ""
                                call["args"] += tc.function.arguments or ""
                except _Retry:
                    round_no -= 1   # this round stalled before saying anything: redo it on the next provider
                    continue
                finally:
                    resp.close()
                if not calls:
                    if carry_ons < MAX_CARRY_ONS and _only_promised("".join(said)):
                        # It said "I'll open your note" and stopped: make it do the work in this same turn.
                        carry_ons += 1
                        messages.append({"role": "assistant", "content": "".join(said)})
                        messages.append({"role": "system", "content": _CARRY_ON.format(
                            how=_CARRY_ON_TOOLS if offered else "")})
                        log.info("model announced an action without taking it; carrying the turn on")
                        think("step", {"id": f"{round_no}.c", "label": "carrying on", "status": "done"})
                        if said and not said[-1].endswith((" ", "\n")):
                            yield " "
                        continue
                    break
                if cancel is not None and cancel.is_set():
                    return
                messages.append({"role": "assistant", "content": "".join(said) or None, "tool_calls": [
                    {"id": c["id"] or f"call_{round_no}_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                    for i, c in enumerate(calls.values())]})
                for i, call in enumerate(calls.values()):
                    try:
                        args = json.loads(call["args"] or "{}")
                    except ValueError:
                        args = {}
                    args = args if isinstance(args, dict) else {}
                    if not answered:   # the model hasn't said anything itself yet: fill the pause
                        line = runner.announce(call["name"], args)
                        if line:
                            chars += len(line)
                            yield line
                    step = {"id": f"{round_no}.{i}", "tool": call["name"]}
                    label, _ = runner.describe(call["name"], args)
                    think("step", {**step, "label": label, "status": "running"})
                    t_tool = time.perf_counter()
                    result = runner.run(call["name"], args)
                    ms = (time.perf_counter() - t_tool) * 1000
                    label, ok = runner.describe(call["name"], args, done=True)
                    think("step", {**step, "label": label, "status": "done" if ok else "failed", "ms": round(ms)})
                    used_tools.append(call["name"])
                    log.info("tool %s ran in %.0fms (%d chars back)%s", call["name"], ms, len(result),
                             "" if ok else " FAILED")
                    messages.append({"role": "tool", "tool_call_id": call["id"] or f"call_{round_no}_{i}",
                                     "content": result})
        except Exception as e:
            log.error("LLM CALL FAILED on %s after %.0fms: %s: %s",
                      route.name, (time.perf_counter() - t0) * 1000, type(e).__name__, e)
            raise LLMError(str(e)) from e
        log.info("llm model=%s effort=%s first_token=%sms total=%.0fms chars=%d%s", route.name, effort,
                 "-" if first_ms is None else f"{first_ms:.0f}", (time.perf_counter() - t0) * 1000, chars,
                 f" tools={','.join(used_tools)}" if used_tools else "")
