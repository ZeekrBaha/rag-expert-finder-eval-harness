import json
from dataclasses import dataclass
from pathlib import Path

KINDS = {"positive", "hard_negative", "abstain"}


@dataclass
class GoldenCase:
    query: str
    kind: str          # positive | hard_negative | abstain
    correct_ids: list[str]
    rationale: str


def load_golden(path) -> list[GoldenCase]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(GoldenCase(d["query"], d["kind"],
                              d.get("correct_ids", []), d.get("rationale", "")))
    return out


def validate_classes(cases: list[GoldenCase]) -> tuple[bool, list[str]]:
    present = {c.kind for c in cases}
    missing = sorted(KINDS - present)
    return (len(missing) == 0, missing)
