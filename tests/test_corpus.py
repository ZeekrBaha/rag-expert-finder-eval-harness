from app.corpus import Record, load_jsonl, save_jsonl


def test_record_roundtrip(tmp_path):
    recs = [
        Record(id="W1", title="Solid-state battery cathodes", abstract="lithium ...",
               author="Jane Li", institution="MIT", concept="batteries", h_index=40),
        Record(id="W2", title="CRISPR delivery", abstract="lipid nanoparticle ...",
               author="Sam Ng", institution="Broad", concept="crispr", h_index=33),
    ]
    p = tmp_path / "c.jsonl"
    save_jsonl(recs, p)
    back = load_jsonl(p)
    assert [r.id for r in back] == ["W1", "W2"]
    assert back[0].author == "Jane Li"
    assert back[1].h_index == 33


def test_load_skips_blank_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"W1","title":"t","abstract":"a","author":"x",'
                 '"institution":"i","concept":"c","h_index":1}\n\n')
    assert len(load_jsonl(p)) == 1
