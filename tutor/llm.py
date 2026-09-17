"""
Groq chat client: streaming, cancellable, with latency logging. Errors are logged loudly and
re-raised as LLMError for the caller to handle; nothing here crashes the app.

If the primary model is rate limited (429), the turn is retried once on LLM_FALLBACK_MODEL, which has
its own separate per-model token allowance on Groq.

Agent mode: with tools, the model's tool calls run through a runner (announce/describe/run) and the conversation
continues in the same stream, for up to LLM_MAX_TOOL_ROUNDS rounds. The runner's short "let me look" line is yielded
while a tool runs, so there is no silent pause. on_think receives the model's reasoning as it streams, a "step"
event before and after each tool, and "answer" when the real reply starts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv
from groq import Groq, RateLimitError

import config

log = logging.getLogger("llm")

ROOT = Path(__file__).resolve().parent.parent
_EFFORT_ORDER = ("low", "medium", "high")


class LLMError(Exception):
    pass


def _deeper(a: str, b: str) -> str:
    return max(a, b, key=lambda e: _EFFORT_ORDER.index(e) if e in _EFFORT_ORDER else 0)


class LLMClient:
    def __init__(self):
        load_dotenv(ROOT / ".env")
        key = os.getenv("GROQ_API_KEY")
        if not key:
            log.error("GROQ_API_KEY missing from .env; every LLM call will fail")
        # Keep the pooled connection alive between turns and open it at startup. Measured: no consistent
        # first-token difference vs httpx's 5s default (both ~130-600ms), so this is cheap insurance, not a fix.
        http_client = httpx.Client(timeout=config.LLM_TIMEOUT_S,
                                   limits=httpx.Limits(keepalive_expiry=config.LLM_KEEPALIVE_S))
        # No SDK retries: a retry after a timeout would blow the latency budget; fail fast instead.
        self._client = Groq(api_key=key or "missing", timeout=config.LLM_TIMEOUT_S, max_retries=0,
                            http_client=http_client)
        self.warmup()

    def warmup(self) -> None:
        """Open the connection before the first real turn. Never raises."""
        t0 = time.perf_counter()
        try:
            self._client.models.list()
            log.info("llm connection warm in %.0fms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            log.error("LLM WARMUP FAILED (%s: %s); first turn may be slow or fail", type(e).__name__, e)

    def _create(self, model: str, messages: list[dict], tools: list[dict] | None, effort: str):
        extra = {}
        if model.startswith("openai/gpt-oss"):
            # Reasoning is generated either way; returning it costs nothing and lets the page show it.
            extra = {"reasoning_effort": effort, "include_reasoning": True}
        options = {"tools": tools} if tools else {}
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS if effort == "low" else config.LLM_MAX_TOKENS_AGENT,
            extra_body=extra,
            **options,
        )

    def stream(self, messages: list[dict], cancel: threading.Event | None = None,
               tools: list[dict] | None = None, runner=None, effort: str | None = None,
               on_think: Callable[[str, dict], None] | None = None) -> Iterator[str]:
        """Yield text deltas. Stops early (closing the HTTP stream) once `cancel` is set.

        runner (with tools): announce(name, args) -> short line or None, describe(name, args, done) -> (label, ok),
        run(name, args) -> result text.
        """
        t0 = time.perf_counter()
        first_ms = None
        chars = 0
        model = config.LLM_MODEL
        effort = effort or config.LLM_REASONING_EFFORT
        messages = list(messages)
        used_tools: list[str] = []
        answered = False

        def think(kind: str, data: dict) -> None:
            if on_think is not None:
                on_think(kind, data)

        try:
            for round_no in range(1, config.LLM_MAX_TOOL_ROUNDS + 2):
                offered = tools if tools and runner is not None and round_no <= config.LLM_MAX_TOOL_ROUNDS else None
                round_effort = effort if round_no == 1 else _deeper(effort, config.LLM_REASONING_EFFORT_AGENT)
                try:
                    resp = self._create(model, messages, offered, round_effort)
                except RateLimitError as e:
                    if not config.LLM_FALLBACK_MODEL or model == config.LLM_FALLBACK_MODEL:
                        raise
                    log.warning("LLM RATE LIMITED on %s (%s); retrying on %s",
                                model, str(e)[:160], config.LLM_FALLBACK_MODEL)
                    model = config.LLM_FALLBACK_MODEL
                    resp = self._create(model, messages, offered, round_effort)
                said: list[str] = []
                calls: dict[int, dict] = {}
                try:
                    for chunk in resp:
                        if cancel is not None and cancel.is_set():
                            log.info("llm cancelled after %.0fms (%d chars)", (time.perf_counter() - t0) * 1000, chars)
                            return
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        reasoning = getattr(choice.delta, "reasoning", None)
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
                finally:
                    resp.close()
                if not calls:
                    break
                if cancel is not None and cancel.is_set():
                    return
                messages.append({"role": "assistant", "content": "".join(said) or None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                    for c in calls.values()]})
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
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        except Exception as e:
            log.error("LLM CALL FAILED on %s after %.0fms: %s: %s",
                      model, (time.perf_counter() - t0) * 1000, type(e).__name__, e)
            raise LLMError(str(e)) from e
        log.info("llm model=%s effort=%s first_token=%sms total=%.0fms chars=%d%s", model, effort,
                 "-" if first_ms is None else f"{first_ms:.0f}", (time.perf_counter() - t0) * 1000, chars,
                 f" tools={','.join(used_tools)}" if used_tools else "")
