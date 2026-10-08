"""Turns a run request (profile name + overrides + optional external outputs) into a
fully explicit, frozen RunSpec that is stored with the run for reproducibility."""
from __future__ import annotations

import itertools
import random
import uuid

from pydantic import BaseModel, Field

from .config import BudgetCfg, GateCfg, PairwiseCfg, Registry


class RunRequest(BaseModel):
    profile: str | None = None
    dataset: str | None = None
    systems: list[str] | None = None
    judges: list[str] | None = None
    rubrics: list[str] | None = None
    pairwise: PairwiseCfg | None = None
    sample: int | None = None
    seed: int | None = None
    bootstrap: int | None = None
    concurrency: int | None = None
    budget: BudgetCfg | None = None
    gate: GateCfg | None = None
    # Outputs produced elsewhere (e.g. by another repo's CI): {system_id: {item_id: response}}
    external_outputs: dict[str, dict[str, str]] | None = None
    label: str | None = None


class RunSpec(BaseModel):
    profile: str | None
    label: str | None = None
    dataset: str
    systems: list[str]
    external_systems: list[str] = Field(default_factory=list)
    judges: list[str]
    rubrics: list[str]  # pinned "id@version"
    pairwise_rubrics: list[str] = Field(default_factory=list)
    pairs: list[tuple[str, str]] = Field(default_factory=list)
    item_ids: list[str]
    seed: int
    bootstrap: int
    concurrency: int
    budget: BudgetCfg
    gate: GateCfg | None = None
    external_outputs: dict[str, dict[str, str]] = Field(default_factory=dict)


class SpecError(ValueError):
    pass


def resolve(reg: Registry, req: RunRequest) -> RunSpec:
    if req.profile and req.profile not in reg.profiles:
        raise SpecError(f"unknown profile {req.profile}")
    base = reg.profiles[req.profile].model_dump() if req.profile else {}

    def pick(name: str, default=None):  # noqa: ANN001, ANN202
        v = getattr(req, name)
        return v if v is not None else base.get(name, default)

    dataset = pick("dataset")
    if not dataset:
        raise SpecError("dataset is required (directly or via a profile)")
    ds = reg.dataset(dataset)

    external = req.external_outputs or {}
    systems = list(pick("systems", []) or [])
    for s in external:
        if s not in systems:
            systems.append(s)
    for s in systems:
        if s not in reg.systems and s not in external:
            raise SpecError(f"unknown system {s} (not in configs/systems and no external outputs given)")
    if len(systems) < 1:
        raise SpecError("at least one system is required")

    judges = list(pick("judges", []) or [])
    for j in judges:
        if j not in reg.judges:
            raise SpecError(f"unknown judge {j}")
    if not judges:
        raise SpecError("at least one judge is required")

    rubrics = [reg.rubric(r).key for r in (pick("rubrics", []) or [])]

    pw = req.pairwise or (PairwiseCfg(**base["pairwise"]) if base.get("pairwise") else PairwiseCfg())
    pw_rubrics = [reg.rubric(r).key for r in pw.rubrics]
    if pw.pairs == "vs_baseline":
        if pw.baseline not in systems:
            raise SpecError("pairwise baseline must be one of the systems")
        pairs = [(pw.baseline, s) for s in systems if s != pw.baseline]
    else:
        pairs = list(itertools.combinations(systems, 2))
    if len(systems) < 2:
        pairs, pw_rubrics = [], []

    item_ids = [it.id for it in ds.items]
    for s, outs in external.items():
        item_ids = [i for i in item_ids if i in outs]
    if not item_ids:
        raise SpecError("no dataset items left (external outputs cover none of the dataset ids)")
    seed = pick("seed", 7)
    sample = pick("sample")
    if sample and sample < len(item_ids):
        item_ids = sorted(random.Random(seed).sample(item_ids, sample))

    gate = req.gate or (GateCfg(**base["gate"]) if base.get("gate") else None)
    if gate:
        for s in (gate.baseline, gate.candidate):
            if s not in systems:
                raise SpecError(f"gate system {s} is not part of the run")

    budget = req.budget or BudgetCfg(**base.get("budget", {}))
    return RunSpec(
        profile=req.profile, label=req.label, dataset=dataset, systems=systems,
        external_systems=[s for s in systems if s in external], judges=judges, rubrics=rubrics,
        pairwise_rubrics=pw_rubrics, pairs=pairs, item_ids=item_ids, seed=seed,
        bootstrap=pick("bootstrap", 1000), concurrency=pick("concurrency", 6), budget=budget,
        gate=gate, external_outputs=external,
    )


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
