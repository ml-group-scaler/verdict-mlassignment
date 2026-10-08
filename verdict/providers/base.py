"""Provider interface. A provider turns (model config, chat messages) into text."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import ModelCfg, ProviderCfg


class ProviderError(RuntimeError):
    """`availability=False` marks errors caused by the reply itself (e.g. empty content),
    which should not trip the circuit breaker the way timeouts / 5xx / 404s do."""

    def __init__(self, message: str, retryable: bool = False, availability: bool = True):
        super().__init__(message)
        self.retryable = retryable
        self.availability = availability


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class Provider(Protocol):
    def complete(self, model: ModelCfg, messages: list[dict]) -> LLMResponse: ...


def build_provider(cfg: ProviderCfg) -> Provider:
    if cfg.type == "openai_compatible":
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    if cfg.type == "mock":
        from .mock import MockProvider

        return MockProvider()
    raise ValueError(f"unknown provider type {cfg.type}")
