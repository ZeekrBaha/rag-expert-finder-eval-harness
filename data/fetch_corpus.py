"""Fetch a small real corpus from OpenAlex across a few research areas.

work_to_record() is the pure, tested transform. fetch_corpus() does the network
call via pyalex and writes data/corpus.jsonl. Run directly:  python -m data.fetch_corpus
"""
from pathlib import Path

from app.corpus import Record, save_jsonl


def _rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def work_to_record(work: dict, h_index: int) -> Record:
    wid = (work.get("id") or "").rsplit("/", 1)[-1]
    authorships = work.get("authorships") or []
    first = authorships[0] if authorships else {}
    author = (first.get("author") or {}).get("display_name", "") or ""
    insts = first.get("institutions") or []
    institution = insts[0]["display_name"] if insts else ""
    concepts = work.get("concepts") or []
    concept = concepts[0]["display_name"] if concepts else ""
    return Record(
        id=wid,
        title=work.get("title") or "",
        abstract=_rebuild_abstract(work.get("abstract_inverted_index")),
        author=author,
        institution=institution,
        concept=concept,
        h_index=h_index,
    )


SEARCH_AREAS = [
    "solid-state battery electrolytes",
    "CRISPR lipid nanoparticle delivery",
    "federated learning privacy",
    "perovskite solar cell stability",
]


def fetch_corpus(per_area: int = 50, out: str = "data/corpus.jsonl") -> int:
    from pyalex import Works  # imported here so tests don't need the network
    records: list[Record] = []
    for area in SEARCH_AREAS:
        works = Works().search(area).get(per_page=per_area)
        for w in works:
            records.append(work_to_record(w, h_index=0))
    save_jsonl(records, Path(out))
    return len(records)


if __name__ == "__main__":
    n = fetch_corpus()
    print(f"wrote {n} records to data/corpus.jsonl")
