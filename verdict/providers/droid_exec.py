"""Factory Droid provider via the official `droid exec` CLI (headless, read-only by default).

Factory documents no raw chat-completions API for `fk-` keys, so each call runs one
non-interactive Droid session in an empty temp directory and reads the JSON result:
    droid exec -o json -m <model> --disable-builtin-skills --cwd <empty dir> -f <prompt file>
The key is passed to the CLI as FACTORY_API_KEY (taken from the provider's api_key_env).
Note: every call carries Droid's agent system prompt (~9k Factory credits/call), so this is
configured as a FALLBACK provider only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..config import ModelCfg, ProviderCfg
from .base import LLMResponse, ProviderError

INSTRUCTION = ("Do not use any tools, do not read or write files. Answer directly with only the text the "
               "conversation below asks for.\n\n")


def flatten(messages: list[dict]) -> str:
    """Droid exec takes one prompt, so a multi-turn exchange (judge reply + repair request) is
    rendered as a transcript."""
    if len(messages) == 1:
        return INSTRUCTION + messages[0]["content"]
    parts = [INSTRUCTION]
    for m in messages:
        role = {"user": "USER", "assistant": "YOUR PREVIOUS REPLY", "system": "SYSTEM"}.get(m["role"], m["role"].upper())
        parts.append(f"### {role}\n{m['content']}\n")
    return "\n".join(parts)


class DroidExecProvider:
    def __init__(self, cfg: ProviderCfg):
        self.cfg = cfg
        self.binary = os.environ.get("DROID_BIN") or shutil.which("droid")
        self.workdir = Path(tempfile.mkdtemp(prefix="verdict-droid-"))  # empty: nothing for the agent to touch

    def complete(self, model: ModelCfg, messages: list[dict]) -> LLMResponse:
        if not self.binary:
            raise ProviderError("droid CLI not found on PATH (install Factory Droid or set DROID_BIN)")
        key = os.environ.get(self.cfg.api_key_env or "", "")
        if not key:
            raise ProviderError(f"{self.cfg.api_key_env} is not set — add it to .env (provider '{self.cfg.name}')")
        env = {**os.environ, "FACTORY_API_KEY": key}
        timeout = model.timeout_s or self.cfg.timeout_s
        retries = self.cfg.max_retries if model.max_retries is None else model.max_retries
        with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=self.workdir, delete=False) as f:
            f.write(flatten(messages))
            prompt_path = f.name
        cmd = [self.binary, "exec", "-o", "json", "-m", model.model, "--disable-builtin-skills",
               "--cwd", str(self.workdir), "-f", prompt_path]
        if model.extra_body.get("reasoning_effort"):
            cmd += ["-r", str(model.extra_body["reasoning_effort"])]
        last: ProviderError | None = None
        try:
            for attempt in range(retries + 1):
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
                except subprocess.TimeoutExpired:
                    last = ProviderError(f"droid exec timed out after {timeout}s ({model.model})", retryable=True)
                except OSError as e:  # binary missing / not executable
                    raise ProviderError(f"cannot run droid CLI at {self.binary}: {e}") from e
                else:
                    latency = (time.perf_counter() - t0) * 1000
                    try:
                        return self._parse(proc, model, latency)
                    except ProviderError as e:
                        last = e
                        if not e.retryable:
                            raise
                if attempt < retries:
                    time.sleep(2 ** attempt)
        finally:
            Path(prompt_path).unlink(missing_ok=True)
        raise last or ProviderError("droid exec failed")

    @staticmethod
    def _parse(proc: subprocess.CompletedProcess, model: ModelCfg, latency_ms: float) -> LLMResponse:
        out = (proc.stdout or "").strip()
        data = None
        for line in reversed(out.splitlines()):  # the JSON result is the last JSON line
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            msg = (proc.stderr or out or "no output").strip()[:300]
            raise ProviderError(f"droid exec ({model.model}) exit {proc.returncode}: {msg}",
                                retryable="Invalid model" not in msg)
        if data.get("is_error") or not str(data.get("result", "")).strip():
            raise ProviderError(f"droid exec ({model.model}) error: {str(data.get('result'))[:300]}", retryable=True)
        u = data.get("usage") or {}
        tin = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        tout = int(u.get("output_tokens") or 0) + int(u.get("thinking_tokens") or 0)
        return LLMResponse(text=str(data["result"]), input_tokens=tin, output_tokens=tout, latency_ms=latency_ms,
                           credits=float(u.get("factory_credits") or 0))
