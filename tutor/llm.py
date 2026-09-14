"""
Groq chat client: streaming, cancellable, with latency logging. Errors are logged loudly and
re-raised as LLMError for the caller to handle; nothing here crashes the app.

If the primary model is rate limited (429), the turn is retried once on LLM_FALLBACK_MODEL, which has
its own separate per-model token allowance on Groq.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv
from groq import Groq, RateLimitError

import config

log = logging.getLogger("llm")

ROOT = Path(__file__).resolve().parent.parent


class LLMError(Exception):
    pass


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

    def _create(self, model: str, messages: list[dict]):
        extra = {}
        if model.startswith("openai/gpt-oss"):
            extra = {"reasoning_effort": config.LLM_REASONING_EFFORT, "include_reasoning": False}
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            extra_body=extra,
        )

    def stream(self, messages: list[dict], cancel: threading.Event | None = None) -> Iterator[str]:
        """Yield text deltas. Stops early (closing the HTTP stream) once `cancel` is set."""
        t0 = time.perf_counter()
        first_ms = None
        chars = 0
        model = config.LLM_MODEL
        try:
            try:
                resp = self._create(model, messages)
            except RateLimitError as e:
                if not config.LLM_FALLBACK_MODEL:
                    raise
                log.warning("LLM RATE LIMITED on %s (%s); retrying on %s",
                            model, str(e)[:160], config.LLM_FALLBACK_MODEL)
                model = config.LLM_FALLBACK_MODEL
                resp = self._create(model, messages)
            try:
                for chunk in resp:
                    if cancel is not None and cancel.is_set():
                        log.info("llm cancelled after %.0fms (%d chars)", (time.perf_counter() - t0) * 1000, chars)
                        return
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        if first_ms is None:
                            first_ms = (time.perf_counter() - t0) * 1000
                        chars += len(delta)
                        yield delta
            finally:
                resp.close()
        except Exception as e:
            log.error("LLM CALL FAILED on %s after %.0fms: %s: %s",
                      model, (time.perf_counter() - t0) * 1000, type(e).__name__, e)
            raise LLMError(str(e)) from e
        log.info("llm model=%s first_token=%sms total=%.0fms chars=%d", model,
                 "-" if first_ms is None else f"{first_ms:.0f}", (time.perf_counter() - t0) * 1000, chars)
