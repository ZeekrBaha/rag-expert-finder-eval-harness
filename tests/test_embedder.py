import numpy as np
from app.embedder import HashingEmbedder, cosine


def test_cosine_basics():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert cosine(a, b) == 1.0
    assert abs(cosine(a, c)) < 1e-9


def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder(dim=64)
    v1 = e.embed("solid state battery electrolyte")
    v2 = e.embed("solid state battery electrolyte")
    assert np.allclose(v1, v2)                       # deterministic
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6      # L2-normalized


def test_shared_words_are_more_similar():
    e = HashingEmbedder(dim=256)
    battery = e.embed("solid state battery electrolyte lithium")
    battery2 = e.embed("lithium battery solid electrolyte cathode")
    crispr = e.embed("crispr cas9 gene editing delivery")
    assert cosine(battery, battery2) > cosine(battery, crispr)


def test_embed_batch_shape():
    e = HashingEmbedder(dim=32)
    m = e.embed_batch(["a b c", "d e f"])
    assert m.shape == (2, 32)
