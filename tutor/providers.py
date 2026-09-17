"""
Where AI text comes from: NVIDIA NIM (OpenAI-compatible, build.nvidia.com) and Groq, tried in order.

routes(kind) lists the (provider, model, client) to try for "chat", "vision" (reading note images) or "summary"
(note titles and topics). With LLM_PROVIDER = "nvidia", NIM comes first and Groq is the backup; a provider whose
key is missing from .env is skipped. Speech (STT and the voice) always uses Groq and isn't routed here.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

import config

log = logging.getLogger("llm")

ROOT = Path(__file__).resolve().parent.parent
_clients: dict[str, object | None] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class Route:
    provider: str      # "nvidia" | "groq"
    model: str
    client: object

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


def client(provider: str):
    """Shared API client for a provider, or None if its key isn't in .env."""
    with _lock:
        if provider in _clients:
            return _clients[provider]
        load_dotenv(ROOT / ".env")
        # Keep pooled connections alive between turns (opened at startup by LLMClient.warmup).
        http = httpx.Client(timeout=config.LLM_TIMEOUT_S, limits=httpx.Limits(keepalive_expiry=config.LLM_KEEPALIVE_S))
        made = None
        if provider == "nvidia":
            key = os.getenv("NVIDIA_API_KEY", "")
            if key and not key.startswith("your_"):
                from openai import OpenAI
                # A short read timeout: NIM sometimes stalls for seconds, and then Groq should answer instead.
                timeout = httpx.Timeout(config.LLM_TIMEOUT_S, connect=3.0, read=config.NVIDIA_STALL_TIMEOUT_S)
                made = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=key, max_retries=0, timeout=timeout,
                              http_client=http)
            else:
                log.warning("NVIDIA_API_KEY missing from .env; using Groq for AI text")
        else:
            key = os.getenv("GROQ_API_KEY")
            if key:
                from groq import Groq
                made = Groq(api_key=key, max_retries=0, timeout=config.LLM_TIMEOUT_S, http_client=http)
            else:
                log.error("GROQ_API_KEY missing from .env")
        _clients[provider] = made
        return made


def routes(kind: str) -> list[Route]:
    order: list[tuple[str, str]] = []
    if config.LLM_PROVIDER == "nvidia":
        nvidia = {"chat": config.NVIDIA_LLM_MODEL, "vision": config.NVIDIA_VISION_MODEL,
                  "summary": config.NVIDIA_SUMMARY_MODEL}[kind]
        order.append(("nvidia", nvidia))
    order += {"chat": [("groq", config.LLM_MODEL), ("groq", config.LLM_FALLBACK_MODEL)],
              "vision": [("groq", config.NOTES_VISION_MODEL)],
              "summary": [("groq", config.NOTES_SUMMARY_MODEL)]}[kind]
    found = []
    for provider, model in order:
        c = client(provider)
        if model and c is not None:
            found.append(Route(provider, model, c))
    return found


def describe(kind: str) -> str:
    """Human-readable provider order, for the privacy panel."""
    names = {"nvidia": "NVIDIA NIM", "groq": "Groq"}
    found = routes(kind)
    if not found:
        return "not set up"
    first = f"{found[0].model} ({names[found[0].provider]})"
    backups = sorted({names[r.provider] for r in found[1:]} - {names[found[0].provider]})
    return first + (f", {' and '.join(backups)} as backup" if backups else "")


def reasoning_options(route: Route, effort: str) -> dict:
    """Request options that turn a model's visible reasoning on, per provider and model family."""
    model = route.model.lower()
    if "gpt-oss" in model:
        if route.provider == "groq":
            return {"extra_body": {"reasoning_effort": effort, "include_reasoning": True}}
        return {"reasoning_effort": effort}
    if "nemotron" in model:
        # Thinking on NIM took ~30s before the first words in testing, so only the "high" effort turns it on.
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": effort == "high"}}}
    return {}


def reasoning_text(delta) -> str | None:
    """Reasoning tokens in a streamed delta (Groq calls them `reasoning`, NIM `reasoning_content`)."""
    return getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)


def status(e: Exception) -> int | None:
    return getattr(e, "status_code", None)
