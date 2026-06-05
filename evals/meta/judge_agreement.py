"""Meta-eval: how well does the LLM judge agree with human golden labels?

agreement_report() is pure + tested. main() wires it to a real judge run later;
for now it loads two label files so it is runnable without network.
"""
import json
import sys

from evals.meta.stats import cohens_kappa, confusion_matrix


def agreement_report(human: list[bool], judge: list[bool]) -> dict:
    if len(human) != len(judge):
        raise ValueError("human and judge label lists must be equal length")
    n = len(human)
    pct = sum(1 for h, j in zip(human, judge) if bool(h) == bool(j)) / n if n else 0.0
    kappa = cohens_kappa([int(bool(x)) for x in human],
                         [int(bool(x)) for x in judge])
    return {
        "n": n,
        "pct_agreement": pct,
        "kappa": kappa,
        "confusion": confusion_matrix(human, judge),
    }


def main(human_path: str, judge_path: str) -> None:
    human = [bool(x) for x in json.loads(open(human_path).read())]
    judge = [bool(x) for x in json.loads(open(judge_path).read())]
    rep = agreement_report(human, judge)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1], sys.argv[2])
