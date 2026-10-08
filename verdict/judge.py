"""LLM-judge core: prompt rendering, robust JSON parsing with one repair attempt,
pointwise scoring, and pairwise comparison with position swapping."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import RubricCfg
from .llm import LLMClient
from .providers import ProviderError
from .settings import ROOT


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(ROOT / "prompts" / "judge"), undefined=StrictUndefined,
                       keep_trailing_newline=True, autoescape=False)


def _repair_text() -> str:
    return (ROOT / "prompts" / "judge" / "repair.txt").read_text().strip()


@dataclass
class JudgeInput:
    """Everything a judge sees about one test case (independent of the dataset format)."""
    question: str
    context: str = ""
    reference: str = ""
    style_guide: str = ""


def _ctx(rubric: RubricCfg, inp: JudgeInput) -> dict:
    return dict(
        rubric=rubric,
        question=inp.question,
        context=inp.context if rubric.uses_context else "",
        reference=inp.reference if rubric.uses_reference else "",
        style_guide=inp.style_guide if rubric.include_style_guide else "",
    )


def render_pointwise(rubric: RubricCfg, inp: JudgeInput, response: str) -> str:
    return _env().get_template("pointwise.j2").render(**_ctx(rubric, inp), response=response)


def render_pairwise(rubric: RubricCfg, inp: JudgeInput, response_a: str, response_b: str) -> str:
    return _env().get_template("pairwise.j2").render(**_ctx(rubric, inp), response_a=response_a, response_b=response_b)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_reply(text: str, required: str) -> dict | None:
    """Extract the last JSON object containing `required` from a model reply.
    Tolerates code fences, leading reasoning text and trailing commentary."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for blob in candidates:
        # scan for balanced {...} spans, last one first
        spans, depth, start = [], 0, None
        in_str, esc = False, False
        for i, ch in enumerate(blob):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append(blob[start: i + 1])
        for s in reversed(spans):
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and required in obj:
                return obj
    return None


@dataclass
class PointResult:
    score: float | None
    rationale: str
    parse_ok: bool
    repaired: bool
    abstain: bool
    error: str | None = None
    served_model: str | None = None  # model that actually produced the verdict (primary or fallback)


@dataclass
class PairResult:
    winner: str | None  # positional: A | B | tie
    rationale: str
    parse_ok: bool
    repaired: bool
    abstain: bool
    error: str | None = None
    served_model: str | None = None


def _chain(models: str | list[str]) -> list[str]:
    return [models] if isinstance(models, str) else list(models)


def _ask_one(client: LLMClient, model_id: str, prompt: str, required: str, validate) -> tuple[dict | None, bool, str | None]:  # noqa: ANN001
    """Call one judge model; on an unparseable/invalid reply, retry once with a repair message.
    Raises ProviderError if the model itself is unavailable (so the caller can fall back)."""
    messages = [{"role": "user", "content": prompt}]
    reply = client.complete(model_id, messages, purpose="judge").text
    obj = parse_json_reply(reply, required)
    if obj is not None and validate(obj):
        return obj, False, None
    messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": _repair_text()}]
    reply2 = client.complete(model_id, messages, purpose="repair").text
    obj = parse_json_reply(reply2, required)
    if obj is not None and validate(obj):
        return obj, True, None
    return None, True, f"unparseable judge reply: {reply2[:200]!r}"


def _ask(client: LLMClient, models: str | list[str], prompt: str, required: str, validate):  # noqa: ANN001, ANN202
    """Try the primary judge model, then each fallback, when a model is unavailable."""
    errors = []
    for model_id in _chain(models):
        try:
            obj, repaired, err = _ask_one(client, model_id, prompt, required, validate)
            return obj, repaired, err, model_id
        except ProviderError as e:
            errors.append(f"{model_id}: {str(e)[:200]}")
    return None, False, " | ".join(errors), None


def judge_pointwise(client: LLMClient, models: str | list[str], rubric: RubricCfg, inp: JudgeInput, response: str) -> PointResult:
    lo, hi = rubric.scale.min, rubric.scale.max

    def valid(o: dict) -> bool:
        try:
            return lo <= int(o["score"]) <= hi
        except (TypeError, ValueError):
            return False

    obj, repaired, err, served = _ask(client, models, render_pointwise(rubric, inp, response), "score", valid)
    if obj is None:
        return PointResult(None, "", parse_ok=False, repaired=repaired, abstain=True, error=err, served_model=served)
    return PointResult(float(int(obj["score"])), str(obj.get("rationale", "")), True, repaired, False, served_model=served)


def judge_pairwise(client: LLMClient, models: str | list[str], rubric: RubricCfg, inp: JudgeInput, a: str, b: str) -> PairResult:
    def valid(o: dict) -> bool:
        return str(o.get("winner", "")).strip().upper() in {"A", "B", "TIE"}

    obj, repaired, err, served = _ask(client, models, render_pairwise(rubric, inp, a, b), "winner", valid)
    if obj is None:
        return PairResult(None, "", parse_ok=False, repaired=repaired, abstain=True, error=err, served_model=served)
    w = str(obj["winner"]).strip().upper()
    return PairResult("tie" if w == "TIE" else w, str(obj.get("rationale", "")), True, repaired, False, served_model=served)


def positional_to_score(winner: str | None, x_is_a: bool) -> float | None:
    """Score for system x (1 win, 0.5 tie, 0 loss) from a positional verdict."""
    if winner is None:
        return None
    if winner == "tie":
        return 0.5
    return 1.0 if (winner == "A") == x_is_a else 0.0


def combine_orders(s1: float | None, s2: float | None) -> tuple[float | None, bool]:
    """Combine x's score from (x as A) and (x as B). Returns (score, flipped).
    A flip is when the two orders name opposite winners — a pure position effect."""
    vals = [s for s in (s1, s2) if s is not None]
    if not vals:
        return None, False
    flipped = s1 is not None and s2 is not None and {s1, s2} == {0.0, 1.0}
    return sum(vals) / len(vals), flipped
