"""Emit a scorecard.json for the eval-ai-ci-gate release gate.

Maps the analyze() report (evals/analyze_run.py) onto the gate's flat
metric_summary schema. Metrics that the run did not produce are null.
"""
import json
import sys
from pathlib import Path

from evals.analyze_run import analyze

SCHEMA_VERSION = "1.0"


def _kind_pass_rate(per_kind: dict, kind: str) -> float | None:
    bucket = per_kind.get(kind)
    if not bucket or not bucket.get("total"):
        return None
    return bucket["passed"] / bucket["total"]


def build_scorecard(report: dict, run_id: str) -> dict:
    """Build the gate scorecard dict from an analyze() report."""
    per_kind = report.get("per_kind", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "metric_summary": {
            "pass_rate_positive": _kind_pass_rate(per_kind, "positive"),
            "pass_rate_hard_negative": _kind_pass_rate(per_kind, "hard_negative"),
            "pass_rate_abstain": _kind_pass_rate(per_kind, "abstain"),
            "judge_kappa": report.get("judge_vs_oracle", {}).get("kappa"),
            "confident_match_precision": report.get("confident_match", {}).get("precision"),
        },
    }


def write_scorecard(report: dict, run_id: str, path: str = "scorecard.json") -> dict:
    """Write the scorecard to `path`; returns the scorecard dict."""
    scorecard = build_scorecard(report, run_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    return scorecard


def main(run_path: str = "results/openai_run.json",
         out: str = "scorecard.json") -> None:
    data = json.load(open(run_path))
    report = analyze(data["results"]["results"])
    run_id = data.get("evalId") or Path(run_path).stem
    scorecard = write_scorecard(report, run_id=str(run_id), path=out)
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(*sys.argv[1:3])
