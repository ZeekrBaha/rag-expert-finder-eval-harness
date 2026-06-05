"""Analyze a Promptfoo results JSON into the meta-eval numbers the project claims.

Computes:
  - per-kind pass rates,
  - judge-vs-oracle agreement (Cohen's kappa + confusion) — validates the LLM judge
    against a deterministic golden oracle,
  - the model's confident-match precision/recall — quantifies the "confident wrong
    match" failure (a committed match that is wrong = a false positive).

normalize_ids() and oracle_correct() are pure + tested; main() reads the JSON.
"""
import json
import sys

from evals.meta.stats import cohens_kappa, confusion_matrix, precision_recall


def normalize_ids(raw) -> set[str]:
    """correct_ids may arrive as a list, a single id string, or '[]'/''. -> set."""
    if isinstance(raw, list):
        return {str(x) for x in raw}
    s = str(raw).strip()
    if s in ("", "[]", "None"):
        return set()
    return {s}


def oracle_correct(kind: str, correct_ids: set[str],
                   expert_id, abstain: bool) -> bool:
    """Deterministic golden verdict for one case."""
    if kind == "abstain":
        return bool(abstain)
    # positive / hard_negative: must commit to a correct id, not abstain.
    return (not abstain) and (expert_id in correct_ids)


def analyze(results: list[dict]) -> dict:
    per_kind_pass: dict[str, list[int]] = {}
    oracle, judge = [], []
    committed_true, committed_pred = [], []  # for confident-match precision/recall
    # per-provider accumulators keyed by provider.label
    prov: dict[str, dict] = {}

    for r in results:
        v = r["vars"]
        out = json.loads(r["response"]["output"])
        kind = v["kind"]
        cids = normalize_ids(v.get("correct_ids"))
        eid = out.get("expert_id")
        abstain = bool(out.get("abstain"))

        oc = oracle_correct(kind, cids, eid, abstain)
        jp = bool(r["success"])
        oracle.append(oc)
        judge.append(jp)

        bucket = per_kind_pass.setdefault(kind, [0, 0])
        bucket[1] += 1
        if jp:
            bucket[0] += 1

        # "committed a match" = not abstain. y_true = that match is correct.
        if not abstain:
            committed_true.append(oc)       # was the committed match actually correct?
            committed_pred.append(True)     # the model asserted a (confident) match

        # --- per-provider breakdown ---
        label = r.get("provider", {}).get("label", "unknown")
        p = prov.setdefault(label, {
            "passed": 0, "total": 0, "commits": 0, "correct_commits": 0,
            "cost_sum": 0.0, "latency_sum": 0.0,
        })
        p["total"] += 1
        if jp:
            p["passed"] += 1
        if not abstain:
            p["commits"] += 1
            if oc:
                p["correct_commits"] += 1
        p["cost_sum"] += r.get("cost") or 0
        p["latency_sum"] += r.get("latencyMs") or 0

    n = len(results)
    pct_agree = sum(1 for o, j in zip(oracle, judge) if o == j) / n if n else 0.0
    kappa = cohens_kappa([int(x) for x in oracle], [int(x) for x in judge])

    # Confident-match precision = correct commits / all commits; recall over positives.
    tp = sum(1 for t in committed_true if t)
    commits = len(committed_true)
    match_precision = tp / commits if commits else 0.0

    by_provider = {}
    for label, p in prov.items():
        total = p["total"]
        c = p["commits"]
        by_provider[label] = {
            "passed": p["passed"],
            "total": total,
            "pass_rate": p["passed"] / total if total else 0.0,
            "commits": c,
            "correct_commits": p["correct_commits"],
            "precision": p["correct_commits"] / c if c else 0.0,
            "avg_cost": p["cost_sum"] / total if total else 0.0,
            "avg_latency_ms": p["latency_sum"] / total if total else 0.0,
        }

    return {
        "n": n,
        "overall_pass": sum(judge) / n if n else 0.0,
        "per_kind": {k: {"passed": v[0], "total": v[1]} for k, v in per_kind_pass.items()},
        "judge_vs_oracle": {
            "kappa": kappa,
            "pct_agreement": pct_agree,
            "confusion": confusion_matrix(oracle, judge),
        },
        "confident_match": {
            "commits": commits,
            "correct_commits": tp,
            "precision": match_precision,
            "wrong_confident_matches": commits - tp,
        },
        "by_provider": by_provider,
    }


def main(path: str = "results/openai_run.json",
         out: str = "results/meta.json") -> None:
    data = json.load(open(path))
    results = data["results"]["results"]
    report = analyze(results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(*sys.argv[1:])
