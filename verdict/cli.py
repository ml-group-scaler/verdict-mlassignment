"""`verdict` command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from . import calibration, db, probes, report
from .config import registry
from .settings import ROOT
from .spec import RunRequest

app = typer.Typer(add_completion=False, help="Verdict — calibrated LLM-as-judge evaluation service")


def _csv(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


@app.command()
def run(
    profile: str = typer.Option("mock", help="configs/profiles/<name>.yaml"),
    sample: int = typer.Option(None, help="evaluate a random subset of N items"),
    systems: str = typer.Option(None, help="comma-separated override"),
    judges: str = typer.Option(None, help="comma-separated override"),
    outputs: Path = typer.Option(None, help="JSON {system_id: {item_id: response}} of externally produced outputs"),
    label: str = typer.Option(None),
    json_out: Path = typer.Option(None, help="write the summary JSON here"),
    md_out: Path = typer.Option(None, help="write the markdown report here"),
    fail_on_gate: bool = typer.Option(False, help="exit 1 if the profile's merge gate fails"),
) -> None:
    """Execute a run synchronously in this process and print the leaderboard."""
    from .runner import run_sync

    db.init()
    req = RunRequest(profile=profile, sample=sample, systems=_csv(systems), judges=_csv(judges), label=label,
                     external_outputs=json.loads(outputs.read_text()) if outputs else None)
    run_id, summary = run_sync(req)
    md = report.run_markdown(summary)
    typer.echo(md)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(summary, indent=2, default=str))
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(md)
    typer.echo(f"run_id={run_id}")
    if fail_on_gate and "gate" in summary and not summary["gate"]["passed"]:
        raise typer.Exit(1)


@app.command()
def show(run_id: str) -> None:
    """Print the markdown report of a stored run."""
    db.init()
    r = db.get_run(run_id)
    if r is None or not r.summary:
        raise typer.BadParameter(f"run {run_id} not found or has no summary")
    typer.echo(report.run_markdown(r.summary))


@app.command()
def calibrate(
    labels: Path = typer.Option(ROOT / "data/human_labels/labels.jsonl", help="human labels JSONL"),
    dataset: str = typer.Option("support_v1"),
    judges: str = typer.Option("glm-5.3,nemotron-3-super"),
    gate_config: Path = typer.Option(ROOT / "configs/calibration_gate.yaml"),
    out: Path = typer.Option(ROOT / "reports/calibration"),
    enforce: bool = typer.Option(False, help="exit 1 if agreement is below the thresholds"),
) -> None:
    """Measure judge agreement with human labels (weighted κ, Spearman, ...)."""
    db.init()
    rows = calibration.load_labels(labels)
    if not rows:
        raise typer.BadParameter(f"no labels in {labels}")
    rep = calibration.calibrate(registry(), dataset, rows, _csv(judges) or [])
    gate_cfg = yaml.safe_load(gate_config.read_text())
    rep["gate"] = calibration.check_gate(rep, gate_cfg["min_weighted_kappa"], gate_cfg.get("judge", "panel"))
    rep["labels_file"] = str(labels)
    probes.save_report("calibration", rep)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps(rep, indent=2))
    md = report.calibration_markdown(rep) + "\n" + ("✅" if rep["gate"]["passed"] else "❌") + \
        " calibration gate: " + ", ".join(f"{c['rubric']} κ={c['kappa'] if c['kappa'] is None else round(c['kappa'], 2)} (≥{c['threshold']})"
                                         for c in rep["gate"]["checks"]) + "\n"
    Path(f"{out}.md").write_text(md)
    typer.echo(md)
    if enforce and not rep["gate"]["passed"]:
        raise typer.Exit(1)


@app.command()
def probe(
    kind: str = typer.Argument(..., help="planted | verbosity | injection"),
    dataset: str = typer.Option("support_v1"),
    judges: str = typer.Option("glm-5.3,nemotron-3-super"),
    run_id: str = typer.Option(None, help="take responses from this run (verbosity/injection)"),
    system: str = typer.Option(None, help="restrict run responses to this system"),
    rubrics: str = typer.Option("faithfulness,helpfulness"),
    limit: int = typer.Option(20),
    out_dir: Path = typer.Option(ROOT / "reports"),
) -> None:
    """Run a judge bias/robustness probe."""
    db.init()
    reg = registry()
    js = _csv(judges) or []
    if kind == "planted":
        rep = probes.planted(reg, dataset, js)
    elif kind in ("verbosity", "injection"):
        if run_id:
            cases = probes.cases_from_run(run_id, system, _csv(rubrics) or [], limit)
        else:
            cases = [{"item_id": p["item_id"], "rubric": r, "response": p["response"]}
                     for p in reg.dataset(dataset).planted for r in (_csv(rubrics) or [])]
        rep = getattr(probes, kind)(reg, dataset, js, cases)
    else:
        raise typer.BadParameter("kind must be planted | verbosity | injection")
    probes.save_report(f"probe_{kind}", rep)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"probe_{kind}.json").write_text(json.dumps(rep, indent=2))
    md = report.probe_markdown(rep)
    (out_dir / f"probe_{kind}.md").write_text(md)
    typer.echo(md)


@app.command("labels-export")
def labels_export(out: Path = typer.Option(ROOT / "data/human_labels/labels.jsonl")) -> None:
    """Export labels collected in the UI (DB) to the versioned JSONL file used by CI."""
    from sqlalchemy import select

    db.init()
    with db.session() as s:
        rows = [h.payload for h in s.scalars(select(db.HumanLabel))]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    typer.echo(f"wrote {len(rows)} labels to {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API (with the background worker)."""
    import uvicorn

    uvicorn.run("verdict.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
