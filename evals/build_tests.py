"""Convert the human golden set into Promptfoo test cases.

golden_to_tests() is pure + tested. main() writes evals/tests.json which
promptfooconfig.yaml references. Keeping golden.jsonl as the source of truth
means human labels and the eval suite never drift.
"""
import json
import sys

from app.golden import load_golden, GoldenCase


def golden_to_tests(cases: list[GoldenCase]) -> list[dict]:
    out = []
    for c in cases:
        out.append({
            "vars": {
                "query": c.query,
                "kind": c.kind,
                "correct_ids": c.correct_ids,
                "rationale": c.rationale,
            }
        })
    return out


def main(golden_path: str = "data/golden.jsonl",
         out_path: str = "evals/tests.json") -> None:
    tests = golden_to_tests(load_golden(golden_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)
    print(f"wrote {len(tests)} tests to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main(*sys.argv[1:])
