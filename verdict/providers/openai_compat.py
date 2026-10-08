"""OpenAI-compatible chat-completions client (NVIDIA NIM, OpenRouter, vLLM, ...).
Retries 429/5xx/timeouts with exponential backoff + jitter and honours Retry-After."""
from __future__ import annotations

import os
import random
import time

import httpx

from ..config import ModelCfg, ProviderCfg
from .base import LLMResponse, ProviderError

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    def __init__(self, cfg: ProviderCfg):
        self.cfg = cfg
        self._client = httpx.Client(base_url=cfg.base_url or "", timeout=cfg.timeout_s)

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.cfg.api_key_env or "", "")
        if not key:
            raise ProviderError(
                f"{self.cfg.api_key_env} is not set — add it to .env (provider '{self.cfg.name}')"
            )
        return {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    def complete(self, model: ModelCfg, messages: list[dict]) -> LLMResponse:
        body: dict = {
            "model": model.model,
            "messages": messages,
            "temperature": model.temperature,
            model.max_tokens_param: model.max_tokens,
            "stream": False,
        }
        if model.top_p is not None:
            body["top_p"] = model.top_p
        body.update(model.extra_body)
        headers = self._headers()
        retries = self.cfg.max_retries if model.max_retries is None else model.max_retries
        timeout = model.timeout_s or self.cfg.timeout_s

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.post("/chat/completions", json=body, headers=headers, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = ProviderError(f"{type(e).__name__}: {e}", retryable=True)
            else:
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    return self._parse(r.json(), latency)
                msg = f"HTTP {r.status_code} from {model.model}: {r.text[:300]}"
                if r.status_code not in RETRYABLE:
                    raise ProviderError(msg)
                last_err = ProviderError(msg, retryable=True)
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    time.sleep(min(float(retry_after), 60))
                    continue
            if attempt < retries:
                time.sleep(min(2 ** attempt + random.random(), 30))
        raise last_err or ProviderError("unknown provider failure")

    @staticmethod
    def _parse(data: dict, latency_ms: float) -> LLMResponse:
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError) as e:
            raise ProviderError(f"malformed response: {str(data)[:300]}") from e
        if not text.strip():
            # Reasoning models can spend the whole token budget thinking.
            raise ProviderError(
                f"empty content (finish_reason={choice.get('finish_reason')}); "
                "raise max_tokens for this model", retryable=False, availability=False,
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )
