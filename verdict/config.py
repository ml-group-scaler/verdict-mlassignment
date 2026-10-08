"""Loads the versioned YAML configs (providers, models, systems, judges, rubrics,
profiles) and datasets, and computes content hashes so every result can be traced
back to the exact configuration that produced it."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .settings import ROOT


def sha(obj: Any) -> str:
    """Stable short hash of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class ProviderCfg(BaseModel):
    name: str
    type: Literal["openai_compatible", "mock"]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60
    max_retries: int = 3


class ModelCfg(BaseModel):
    id: str
    provider: str
    model: str
    family: str
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 1024
    max_tokens_param: str = "max_tokens"  # some endpoints expect "max_completion_tokens"
    timeout_s: float | None = None  # overrides the provider default
    max_retries: int | None = None
    rpm: int | None = None
    max_concurrency: int | None = None  # in-flight cap, so one stalled endpoint can't hog every worker thread
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    behaviour: dict[str, Any] = Field(default_factory=dict)  # mock models only


class SystemCfg(BaseModel):
    id: str
    description: str = ""
    type: Literal["model", "external"] = "model"
    model: str | None = None
    system_prompt: str | None = None
    include_policy: bool = True

    def prompt_text(self) -> str:
        return (ROOT / self.system_prompt).read_text() if self.system_prompt else ""


class JudgeCfg(BaseModel):
    id: str
    description: str = ""
    model: str
    # Tried in order when the primary model errors / times out / has an open circuit.
    fallback_models: list[str] = Field(default_factory=list)


class Scale(BaseModel):
    min: int = 1
    max: int = 5


class RubricCfg(BaseModel):
    id: str
    version: int
    name: str
    description: str
    criteria: str = ""
    uses_context: bool = True
    uses_reference: bool = True
    include_style_guide: bool = False
    scale: Scale = Field(default_factory=Scale)
    anchors: dict[int, str] = Field(default_factory=dict)
    programmatic_checks: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def hash(self) -> str:
        return sha(self.model_dump())


class PairwiseCfg(BaseModel):
    rubrics: list[str] = Field(default_factory=list)
    pairs: Literal["all", "vs_baseline"] = "all"
    baseline: str | None = None


class GateCfg(BaseModel):
    baseline: str
    candidate: str
    mode: Literal["noninferiority", "significant_regression"] = "noninferiority"
    margin: float = 0.3
    rubrics: list[str] = Field(default_factory=list)
    min_scores: dict[str, float] = Field(default_factory=dict)


class BudgetCfg(BaseModel):
    max_calls: int = 5000
    max_usd: float = 5.0


class ProfileCfg(BaseModel):
    id: str
    dataset: str
    systems: list[str]
    judges: list[str]
    rubrics: list[str]
    pairwise: PairwiseCfg = Field(default_factory=PairwiseCfg)
    sample: int | None = None
    seed: int = 7
    bootstrap: int = 1000
    concurrency: int = 6
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    gate: GateCfg | None = None


class Item(BaseModel):
    id: str
    category: str = ""
    difficulty: str = ""
    question: str
    context_sections: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    id: str
    status: str = ""
    description: str = ""
    items: list[Item]
    policy_text: str
    sections: dict[str, str]
    style_guide: str = ""
    planted: list[dict[str, Any]] = Field(default_factory=list)
    hash: str

    def item(self, item_id: str) -> Item:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)

    def context_for(self, item: Item) -> str:
        secs = item.context_sections or list(self.sections)
        return "\n\n".join(self.sections[s] for s in secs if s in self.sections)

    @staticmethod
    def reference_for(item: Item) -> str:
        lines = ["Key facts a good answer covers:"] + [f"- {f}" for f in item.key_facts]
        if item.must_not:
            lines += ["The answer must NOT:"] + [f"- {m}" for m in item.must_not]
        return "\n".join(lines)


def parse_policy_sections(text: str) -> dict[str, str]:
    """Split the policy markdown on '## S<n> — Title' headings."""
    sections: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(S\d+)\b", line)
        if m:
            current = m.group(1)
            sections[current] = line.lstrip("# ").strip()
        elif current:
            sections[current] += "\n" + line
    return {k: v.strip() for k, v in sections.items()}


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class Registry:
    """All configs in one place. Rubrics are addressable as 'id' (latest version) or 'id@N'."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        cfg = root / "configs"
        self.providers = {
            name: ProviderCfg(name=name, **v)
            for name, v in _load_yaml(cfg / "providers.yaml")["providers"].items()
        }
        self.models = {
            mid: ModelCfg(id=mid, **v) for mid, v in _load_yaml(cfg / "models.yaml")["models"].items()
        }
        self.systems = {s.id: s for s in (SystemCfg(**_load_yaml(p)) for p in sorted((cfg / "systems").glob("*.yaml")))}
        self.judges = {j.id: j for j in (JudgeCfg(**_load_yaml(p)) for p in sorted((cfg / "judges").glob("*.yaml")))}
        self.rubric_versions: dict[str, dict[int, RubricCfg]] = {}
        for p in sorted((cfg / "rubrics").glob("*.yaml")):
            r = RubricCfg(**_load_yaml(p))
            self.rubric_versions.setdefault(r.id, {})[r.version] = r
        self.profiles = {p.stem: ProfileCfg(**_load_yaml(p)) for p in sorted((cfg / "profiles").glob("*.yaml"))}
        self._datasets: dict[str, Dataset] = {}

    # -- lookups -------------------------------------------------------------
    def rubric(self, ref: str) -> RubricCfg:
        rid, _, ver = ref.partition("@")
        versions = self.rubric_versions[rid]
        return versions[int(ver)] if ver else versions[max(versions)]

    def model_for_judge(self, judge_id: str) -> ModelCfg:
        return self.models[self.judges[judge_id].model]

    def judge_chain(self, judge_id: str) -> list[str]:
        """Primary model id followed by fallback model ids."""
        j = self.judges[judge_id]
        return [j.model] + [m for m in j.fallback_models if m != j.model]

    def family_of_system(self, system_id: str) -> str | None:
        s = self.systems.get(system_id)
        return self.models[s.model].family if s and s.model else None

    def dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self._datasets:
            meta_path = self.root / "data" / "datasets" / f"{dataset_id}.meta.yaml"
            meta = _load_yaml(meta_path)
            items = [Item(**d) for d in _read_jsonl(self.root / meta["items"])]
            policy = (self.root / meta["policy"]).read_text()
            sections = parse_policy_sections(policy)
            planted = _read_jsonl(self.root / meta["planted"]) if meta.get("planted") else []
            self._datasets[dataset_id] = Dataset(
                id=dataset_id,
                status=meta.get("status", ""),
                description=meta.get("description", ""),
                items=items,
                policy_text=policy,
                sections=sections,
                style_guide=sections.get(meta.get("style_guide_section", ""), ""),
                planted=planted,
                hash=sha([i.model_dump() for i in items] + [policy]),
            )
        return self._datasets[dataset_id]

    def config_hash(self, *, dataset: str, systems: list[str], judges: list[str], rubrics: list[str],
                    external: dict[str, dict[str, str]] | None = None) -> str:
        """Hash of every config + prompt file that can influence a run's numbers.
        External systems are hashed by their supplied outputs."""
        external = external or {}
        parts: dict[str, Any] = {"dataset": self.dataset(dataset).hash, "systems": {}, "judges": {}, "rubrics": {}}
        for s in systems:
            sc = self.systems.get(s)
            if s in external:
                parts["systems"][s] = sha(external[s])
                continue
            if sc is None or sc.type == "external":
                parts["systems"][s] = "external"
                continue
            parts["systems"][s] = [sc.model_dump(), self.models[sc.model].model_dump(), sc.prompt_text()]
        for j in judges:
            parts["judges"][j] = self.model_for_judge(j).model_dump()
        for r in rubrics:
            parts["rubrics"][r] = self.rubric(r).hash
        for t in ("pointwise.j2", "pairwise.j2", "repair.txt"):
            parts[t] = (self.root / "prompts" / "judge" / t).read_text()
        return sha(parts)


@lru_cache(maxsize=1)
def registry() -> Registry:
    return Registry()
