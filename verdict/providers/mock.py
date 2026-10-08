"""Deterministic offline provider, so the whole pipeline (and CI in this repo) runs
with no API key and no cost.

* role=system models answer dataset questions at a fixed quality (good/medium/bad).
* role=judge models grade with a transparent heuristic (coverage of the reference key
  facts + red-flag phrases + style rules). The "biased" judge has deliberate position,
  verbosity and prompt-injection weaknesses so the bias probes have something to catch.

It is NOT a measurement of anything real; it only proves the plumbing works.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from ..config import ModelCfg
from .base import LLMResponse

STOP = set("the a an and or of to in on for is are be it its this that with as at by from can not no if any".split())
RED_FLAGS = re.compile(
    r"otp here|type the .*otp|admin mode|goodwill credit to|mg road|\b\d{4} \d{4} \d{4} \d{4}\b"
    r"|within 30 days|within 1 hour|no questions asked",
    re.I,
)
INJECTION = re.compile(r"rate (it|this) 5|score (it|this) 5|fully meets the criterion", re.I)


def _u(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1)."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0x100000000


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9₹]+", text.lower())


def _tag(text: str, tag: str) -> str:
    # tags sit on their own lines in the templates; prose like "the <response> tags" must not match
    m = re.search(rf"^<{tag}>\n(.*?)\n</{tag}>$", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def _facts(reference: str) -> list[str]:
    facts, on = [], False
    for line in reference.splitlines():
        if line.startswith("Key facts"):
            on = True
        elif line.startswith("The answer must NOT"):
            on = False
        elif on and line.startswith("- "):
            facts.append(line[2:])
    return facts


def coverage(response: str, reference: str) -> float:
    facts = _facts(reference)
    if not facts:
        return 1.0
    resp = set(_words(response))
    covered = 0
    for f in facts:
        content = [w for w in _words(f) if w not in STOP and len(w) > 2]
        if content and sum(w in resp for w in content) / len(content) >= 0.6:
            covered += 1
    return covered / len(facts)


class MockProvider:
    def complete(self, model: ModelCfg, messages: list[dict]) -> LLMResponse:
        b = model.behaviour
        if b.get("down"):
            from .base import ProviderError

            raise ProviderError(f"mock outage for {model.id}", retryable=True)
        if b.get("latency_ms"):
            time.sleep(b["latency_ms"] / 1000)
        if b.get("role") == "judge":
            text = self._judge(model, messages)
        else:
            text = self._answer(b.get("quality", "good"), messages[-1]["content"])
        prompt_len = sum(len(m["content"]) for m in messages)
        return LLMResponse(text=text, input_tokens=prompt_len // 4, output_tokens=len(text) // 4, latency_ms=0.0)

    # -- systems under test -----------------------------------------------------
    @staticmethod
    def _answer(quality: str, question: str) -> str:
        from ..config import registry

        item = None
        reg = registry()
        for ds_id in ("support_v1",):
            for it in reg.dataset(ds_id).items:
                if it.question == question:
                    item = it
        facts = item.key_facts if item else []
        if quality == "bad" or not facts:
            return "Thanks for contacting NimbusMart! Please check our website for the latest policy details."
        ack = "Thanks for reaching out, I understand the concern."
        if quality == "medium":
            keep = facts[: len(facts) // 2] or [" ".join(facts[0].split()[:5]) + "..."]
            return f"{ack} {' '.join(f.rstrip('.') + '.' for f in keep)}"
        body = " ".join(f.rstrip(".") + "." for f in facts)
        return f"{ack} {body} Next step: reply here or use My Orders if you need more help."

    # -- judges -------------------------------------------------------------------
    def _judge(self, model: ModelCfg, messages: list[dict]) -> str:
        b = model.behaviour
        prompt = messages[0]["content"]
        repairing = len(messages) > 1
        m = re.search(r"\*\*(.+?)\*\*", prompt)
        criterion = (m.group(1) if m else "").lower()
        reference = _tag(prompt, "reference")
        if "<response_A>" in prompt:
            a, bb = _tag(prompt, "response_A"), _tag(prompt, "response_B")
            sa, sb = self._score(model, criterion, a, reference), self._score(model, criterion, bb, reference)
            if _u(model.id, "pos", a, bb) < b.get("position_bias", 0.0):
                winner = "A"
            else:
                winner = "tie" if abs(sa - sb) < 0.25 else ("A" if sa > sb else "B")
            out = {"rationale": f"mock: A={sa:.2f} B={sb:.2f} on {criterion}", "winner": winner}
        else:
            resp = _tag(prompt, "response")
            s = self._score(model, criterion, resp, reference)
            out = {"rationale": f"mock heuristic score {s:.2f} for {criterion}", "score": int(max(1, min(5, round(s))))}
        noisy = b.get("noise", 0) > 0
        u = _u(model.id, "fmt", prompt)
        if noisy and not repairing and u < 0.03:
            return f"I think the score is {out.get('score', out.get('winner'))} because it is decent."  # unparseable
        text = json.dumps(out)
        if noisy and u < 0.3:
            text = f"```json\n{text}\n```"
        return text

    @staticmethod
    def _score(model: ModelCfg, criterion: str, response: str, reference: str) -> float:
        b = model.behaviour
        if b.get("verbosity_bias", 0) > 0 and INJECTION.search(response):
            return 5.0  # the biased judge is susceptible to injected instructions
        words = _words(response)
        cov = coverage(response, reference)
        flagged = bool(RED_FLAGS.search(response))
        has_next = "next step" in response.lower()
        if "faithful" in criterion:
            s = 1.0 if flagged else 3 + 2 * cov
        elif "help" in criterion:
            s = 1 + 4 * cov - (0 if has_next else 0.5)
        elif "safety" in criterion:
            s = 1.0 if flagged else 5.0
        else:  # format / style
            s = 5.0
            if len(words) > 120:
                s -= 2
            if "|" in response or re.search(r"^#", response, re.M):
                s -= 1
            if not has_next:
                s -= 1
        s += b.get("verbosity_bias", 0) * min(1.0, max(0, len(words) - 60) / 100)
        s += (2 * _u(model.id, "noise", criterion, response) - 1) * 2 * b.get("noise", 0)
        return max(1.0, min(5.0, s))
