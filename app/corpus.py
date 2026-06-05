import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Record:
    id: str
    title: str
    abstract: str
    author: str
    institution: str
    concept: str
    h_index: int

    def text(self) -> str:
        """Concatenated field used for embedding/retrieval."""
        return f"{self.title}. {self.abstract}"


def save_jsonl(records: list[Record], path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_jsonl(path) -> list[Record]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(Record(**json.loads(line)))
    return out
