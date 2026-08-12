"""Result stores: in memory, or lazily from .npz."""

import numpy as np
import pytest

from bemkit.results.store import MemoryStore, NpzStore

DATA = {"pEvalGrid": np.ones((3, 2), dtype=complex),
        "pBoundary": np.zeros((4, 2), dtype=complex)}


def test_memory_store_returns_what_it_was_given():
    s = MemoryStore(DATA)
    np.testing.assert_array_equal(s.get("pEvalGrid"), DATA["pEvalGrid"])


def test_memory_store_missing_quantity_raises():
    with pytest.raises(KeyError, match="vEvalGrid"):
        MemoryStore(DATA).get("vEvalGrid")


def test_npz_round_trip(tmp_path):
    p = tmp_path / "r.npz"
    MemoryStore(DATA).save(p)
    s = NpzStore(p)
    np.testing.assert_array_equal(s.get("pEvalGrid"), DATA["pEvalGrid"])


def test_npz_is_lazy_and_cached(tmp_path):
    p = tmp_path / "r.npz"
    MemoryStore(DATA).save(p)
    s = NpzStore(p)
    assert s.get("pBoundary") is s.get("pBoundary")   # cached, not re-read


def test_available_lists_quantities(tmp_path):
    p = tmp_path / "r.npz"
    MemoryStore(DATA).save(p)
    assert set(NpzStore(p).available()) == {"pEvalGrid", "pBoundary"}
