from data.fetch_corpus import work_to_record


OA_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Federated learning for edge devices",
    "abstract_inverted_index": {"Federated": [0], "learning": [1], "rocks": [2]},
    "authorships": [{
        "author": {"display_name": "Ada Lovelace"},
        "institutions": [{"display_name": "Cambridge"}],
    }],
    "concepts": [{"display_name": "Federated learning"}, {"display_name": "ML"}],
}


def test_work_to_record_rebuilds_abstract_and_picks_first_author():
    r = work_to_record(OA_WORK, h_index=12)
    assert r.id == "W123"
    assert r.title.startswith("Federated learning")
    assert r.abstract == "Federated learning rocks"   # inverted index reconstructed
    assert r.author == "Ada Lovelace"
    assert r.institution == "Cambridge"
    assert r.concept == "Federated learning"
    assert r.h_index == 12


def test_work_to_record_handles_missing_fields():
    r = work_to_record({"id": "https://openalex.org/W9", "title": None}, h_index=0)
    assert r.id == "W9"
    assert r.title == ""
    assert r.abstract == ""
    assert r.author == ""
