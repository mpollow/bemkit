"""Results: access to solved eval-grid/boundary quantities over a Store."""

import numpy as np
import pytest

from bemkit.results.results import Results
from bemkit.results.store import MemoryStore

FREQS = np.array([250.0, 500.0, 750.0])

# Two field-point sets, "a" (2 points) then "b" (3 points), concatenated in
# that order into a 5-row eval-grid array — mirrors what
# bemkit.solvers.fieldpoints.combine would hand to a Results.
P_EVAL = np.arange(5 * 3, dtype=complex).reshape(5, 3)
V_EVAL = (np.arange(5 * 3 * 3, dtype=complex)).reshape(5, 3, 3)
P_BND = np.arange(4 * 3, dtype=complex).reshape(4, 3) * 1j
V_BND = np.arange(4 * 3, dtype=complex).reshape(4, 3) * 2j

OFFSETS = {"a": np.array([0, 1]), "b": np.array([2, 3, 4])}


def _store():
    return MemoryStore({
        "pEvalGrid": P_EVAL.copy(),
        "vEvalGrid": V_EVAL.copy(),
        "pBoundary": P_BND.copy(),
        "vBoundary": V_BND.copy(),
    })


def _results():
    return Results(_store(), FREQS, OFFSETS)


def test_pressure_splits_by_set():
    r = _results()
    np.testing.assert_array_equal(r.pressure("a"), P_EVAL[[0, 1]])
    np.testing.assert_array_equal(r.pressure("b"), P_EVAL[[2, 3, 4]])


def test_velocity_splits_by_set():
    r = _results()
    np.testing.assert_array_equal(r.velocity("a"), V_EVAL[[0, 1]])
    np.testing.assert_array_equal(r.velocity("b"), V_EVAL[[2, 3, 4]])


def test_pressure_with_no_name_returns_everything():
    r = _results()
    np.testing.assert_array_equal(r.pressure(), P_EVAL)
    np.testing.assert_array_equal(r.pressure(None), P_EVAL)


def test_unknown_set_names_the_available_ones():
    r = _results()
    with pytest.raises(KeyError, match="a.*b|b.*a"):
        r.pressure("nope")


def test_boundary_quantities():
    r = _results()
    np.testing.assert_array_equal(r.boundary_pressure(), P_BND)
    np.testing.assert_array_equal(r.boundary_velocity(), V_BND)


def test_set_names_lists_every_offset_key():
    assert _results().set_names == ["a", "b"]


# -- scaled_by ---------------------------------------------------------------

def test_scaled_by_multiplies_per_frequency():
    r = _results()
    spectrum = np.array([2.0, 3.0, 4.0]) + 0j
    scaled = r.scaled_by(spectrum)
    expected = P_EVAL * spectrum[None, :]
    np.testing.assert_allclose(scaled.pressure(), expected)
    # original is untouched -- scaled_by returns a new view
    np.testing.assert_array_equal(r.pressure(), P_EVAL)


def test_scaled_by_composes():
    r = _results()
    a = np.array([2.0, 1.0, 1.0]) + 0j
    b = np.array([1.0, 3.0, 1.0]) + 0j
    twice = r.scaled_by(a).scaled_by(b)
    once = r.scaled_by(a * b)
    np.testing.assert_allclose(twice.pressure(), once.pressure())


def test_scaled_by_applies_to_velocity_and_boundary_too():
    r = _results()
    spectrum = np.array([1.0, 2.0, 1.0]) + 0j
    scaled = r.scaled_by(spectrum)
    np.testing.assert_allclose(scaled.velocity(), V_EVAL * spectrum[None, :, None])
    np.testing.assert_allclose(scaled.boundary_pressure(), P_BND * spectrum[None, :])
    np.testing.assert_allclose(scaled.boundary_velocity(), V_BND * spectrum[None, :])


def test_scaled_by_rejects_length_mismatch():
    r = _results()
    with pytest.raises(ValueError, match="3"):
        r.scaled_by(np.array([1.0, 2.0]))


# -- materialise ---------------------------------------------------------------

def test_materialise_bakes_in_scale_and_is_idempotent_to_read():
    r = _results().scaled_by(np.array([2.0, 2.0, 2.0]) + 0j)
    before = r.pressure()
    r.materialise()
    np.testing.assert_allclose(r.pressure(), before)
    assert r.scale is None
    assert isinstance(r.store, MemoryStore)


# -- save / load ---------------------------------------------------------------

def test_save_load_round_trips_frequencies_and_offsets(tmp_path):
    r = _results()
    r.save(tmp_path / "results.npz")
    loaded = Results.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(loaded.frequencies, FREQS)
    assert loaded.set_names == ["a", "b"]
    np.testing.assert_array_equal(loaded.offsets["a"], OFFSETS["a"])
    np.testing.assert_array_equal(loaded.offsets["b"], OFFSETS["b"])
    np.testing.assert_allclose(loaded.pressure(), P_EVAL)
    np.testing.assert_allclose(loaded.boundary_pressure(), P_BND)


def test_save_load_round_trips_arrays_exactly(tmp_path):
    r = _results()
    r.save(tmp_path / "results.npz")
    loaded = Results.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(loaded.pressure("b"), P_EVAL[[2, 3, 4]])


def test_save_accepts_a_directory(tmp_path):
    r = _results()
    r.save(tmp_path)
    assert (tmp_path / "results.npz").exists()


def test_load_accepts_a_directory(tmp_path):
    r = _results()
    r.save(tmp_path)
    loaded = Results.load(tmp_path)
    np.testing.assert_allclose(loaded.pressure(), P_EVAL)


def test_save_persists_the_effective_scaled_values(tmp_path):
    spectrum = np.array([2.0, 1.0, 1.0]) + 0j
    r = _results().scaled_by(spectrum)
    r.save(tmp_path / "results.npz")
    loaded = Results.load(tmp_path / "results.npz")
    np.testing.assert_allclose(loaded.pressure(), P_EVAL * spectrum[None, :])
