"""Single gateway for every LLM call: content-hash cache, per-model rate limiting,
run budget enforcement, and per-call telemetry (latency, tokens, cost)."""
from __future__ import annotations

import threading
import time

from . import db
from .config import Registry, sha
from .providers import LLMResponse, Provider, ProviderError, build_provider


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, max_calls: int, max_usd: float):
        self.max_calls, self.max_usd = max_calls, max_usd
        self.calls, self.usd = 0, 0.0
        self._lock = threading.Lock()

    def reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"call budget of {self.max_calls} exhausted")
            if self.max_usd > 0 and self.usd >= self.max_usd:
                raise BudgetExceeded(f"cost budget of ${self.max_usd:.2f} exhausted")
            self.calls += 1

    def spend(self, usd: float) -> None:
        with self._lock:
            self.usd += usd


class RateLimiter:
    """Spaces requests to at most `rpm` per minute (per model, process-wide)."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


class CircuitBreaker:
    """Stops calling a model that keeps failing: after `threshold` consecutive failures the
    circuit opens for `cooldown_s`, during which calls fail instantly (so the judge falls back
    immediately instead of waiting out timeouts). One trial call is allowed after the cooldown."""

    def __init__(self, threshold: int = 2, cooldown_s: float = 300):
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.failures, self.open_until = 0, 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            return time.monotonic() >= self.open_until

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self.failures, self.open_until = 0, 0.0
                return
            self.failures += 1
            if self.failures >= self.threshold:
                self.open_until = time.monotonic() + self.cooldown_s


_providers: dict[str, Provider] = {}
_breakers: dict[str, CircuitBreaker] = {}
_semaphores: dict[str, threading.BoundedSemaphore] = {}
_limiters: dict[str, RateLimiter] = {}
_global_lock = threading.Lock()


class LLMClient:
    def __init__(self, reg: Registry, run_id: str | None = None, budget: Budget | None = None, use_cache: bool = True):
        self.reg, self.run_id, self.budget, self.use_cache = reg, run_id, budget, use_cache

    def _provider(self, name: str) -> Provider:
        with _global_lock:
            if name not in _providers:
                _providers[name] = build_provider(self.reg.providers[name])
            return _providers[name]

    def _limiter(self, model_id: str, rpm: int | None) -> RateLimiter | None:
        if not rpm:
            return None
        with _global_lock:
            return _limiters.setdefault(model_id, RateLimiter(rpm))

    def _breaker(self, model_id: str) -> CircuitBreaker:
        with _global_lock:
            return _breakers.setdefault(model_id, CircuitBreaker())

    def cache_key(self, model_id: str, messages: list[dict]) -> str:
        m = self.reg.models[model_id]
        p = self.reg.providers[m.provider]
        return sha([p.base_url, m.model, m.temperature, m.top_p, m.max_tokens, m.extra_body, m.behaviour, messages])

    def complete(self, model_id: str, messages: list[dict], purpose: str) -> LLMResponse:
        m = self.reg.models[model_id]
        key = self.cache_key(model_id, messages)
        if self.use_cache:
            t0 = time.perf_counter()
            with db.session() as s:
                hit = s.get(db.CacheEntry, key)
            if hit is not None:
                lat = (time.perf_counter() - t0) * 1000
                self._record(purpose, model_id, lat, hit.input_tokens, hit.output_tokens, 0.0, True, None)
                return LLMResponse(hit.text, hit.input_tokens, hit.output_tokens, lat, cached=True)

        breaker = self._breaker(model_id)
        if not breaker.allow():
            self._record(purpose, model_id, 0.0, 0, 0, 0.0, False, "circuit open (model failing repeatedly)")
            raise ProviderError(f"circuit open for {model_id}", retryable=True)
        if self.budget:
            self.budget.reserve()
        limiter = self._limiter(model_id, m.rpm)
        if limiter:
            limiter.wait()
        sem = None
        if m.max_concurrency:
            with _global_lock:
                sem = _semaphores.setdefault(model_id, threading.BoundedSemaphore(m.max_concurrency))
        t0 = time.perf_counter()
        try:
            if sem:
                sem.acquire()
                if not breaker.allow():  # the circuit may have opened while we queued
                    raise ProviderError(f"circuit open for {model_id}", retryable=True)
            try:
                resp = self._provider(m.provider).complete(m, messages)
            finally:
                if sem:
                    sem.release()
        except ProviderError as e:
            if e.availability:
                breaker.record(False)
            self._record(purpose, model_id, (time.perf_counter() - t0) * 1000, 0, 0, 0.0, False, str(e)[:500])
            raise
        breaker.record(True)
        latency = (time.perf_counter() - t0) * 1000
        cost = resp.input_tokens * m.price_in_per_mtok / 1e6 + resp.output_tokens * m.price_out_per_mtok / 1e6
        if self.budget:
            self.budget.spend(cost)
        if self.use_cache:
            with db.write() as s:
                s.merge(db.CacheEntry(key=key, model_id=model_id, text=resp.text,
                                      input_tokens=resp.input_tokens, output_tokens=resp.output_tokens))
        self._record(purpose, model_id, latency, resp.input_tokens, resp.output_tokens, cost, False, None)
        resp.latency_ms = latency
        return resp

    def _record(self, purpose, model_id, latency, tin, tout, cost, hit, err) -> None:  # noqa: ANN001
        db.add_all([db.LLMCall(run_id=self.run_id, purpose=purpose, model_id=model_id, latency_ms=latency,
                               input_tokens=tin, output_tokens=tout, cost_usd=cost, cache_hit=hit, error=err)])
