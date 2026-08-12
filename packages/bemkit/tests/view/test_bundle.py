"""The viewer's data plumbing: which bins exist, which one a slider lands on,
and where a bundle's Results come from.

All of it headless. The rendering is not testable here; this is the part that
decides *what* gets rendered, and it is where a partial band or a stale
results.npz turns into a wrong picture rather than an error.
"""

import numpy as np
import pytest

from bemkit.results.results import Results
from bemkit.results.store import MemoryStore
from bemkit.view.bundle import computed_bins, load_results, nearest_bin

FREQS = np.array([250.0, 500.0, 750.0, 1000.0, 1250.0])


def _results(p_eval=None, p_bnd=None):
    arrays = {}
    if p_eval is not None:
        arrays["pEvalGrid"] = p_eval
    if p_bnd is not None:
        arrays["pBoundary"] = p_bnd
    return Results(MemoryStore(arrays), FREQS, {"a": np.array([0, 1])})


class TestComputedBins:
    """A partial run leaves the other bins NaN — ``load_bins`` fills them that
    way so a missing bin is distinguishable from one that solved to zero. The
    viewer must never offer them: NaN reaches the colour mapper as "below the
    range" and paints a confident, wrong colour.
    """

    def test_all_bins_when_everything_is_finite(self):
        r = _results(p_eval=np.ones((2, 5), dtype=complex))
        np.testing.assert_array_equal(computed_bins(r), [0, 1, 2, 3, 4])

    def test_nan_bins_are_excluded(self):
        p = np.ones((2, 5), dtype=complex)
        p[:, [1, 3, 4]] = np.nan
        np.testing.assert_array_equal(computed_bins(_results(p_eval=p)),
                                      [0, 2])

    def test_a_bin_that_solved_to_zero_still_counts(self):
        """The distinction the NaN fill exists for. Testing finiteness rather
        than truthiness is what keeps a genuine null from disappearing."""
        p = np.zeros((2, 5), dtype=complex)
        np.testing.assert_array_equal(computed_bins(_results(p_eval=p)),
                                      [0, 1, 2, 3, 4])

    def test_a_bin_counts_when_any_quantity_has_it(self):
        """Eval and boundary are written per bin by the same solve, but a
        bundle can hold one and not the other; the union is what is drawable.
        """
        pe = np.full((2, 5), np.nan, dtype=complex)
        pe[:, 0] = 1
        pb = np.full((3, 5), np.nan, dtype=complex)
        pb[:, 4] = 1
        np.testing.assert_array_equal(
            computed_bins(_results(p_eval=pe, p_bnd=pb)), [0, 4])

    def test_no_results_at_all_is_empty_not_an_error(self):
        assert len(computed_bins(_results())) == 0
        p = np.full((2, 5), np.nan, dtype=complex)
        assert len(computed_bins(_results(p_eval=p))) == 0

    def test_it_works_on_a_three_vector_quantity(self):
        """vEvalGrid is (N, nf, 3): the frequency axis is 1, not the last
        one. Reducing over the wrong axes gives an array of the wrong length
        and would silently mis-map bins to frequencies."""
        v = np.ones((2, 5, 3), dtype=complex)
        v[:, [0, 2], :] = np.nan
        r = Results(MemoryStore({"vEvalGrid": v}), FREQS,
                    {"a": np.array([0, 1])})
        np.testing.assert_array_equal(
            computed_bins(r, quantities=("vEvalGrid",)), [1, 3, 4])

    def test_a_partially_nan_bin_still_counts(self):
        """NumCalc writes a bin's rows incrementally; a half-written bin is
        caught by the reader, not here. Requiring *every* row finite would
        drop a legitimately zero-padded quantity."""
        p = np.ones((4, 5), dtype=complex)
        p[2, 1] = np.nan
        assert 1 in computed_bins(_results(p_eval=p))


class TestNearestBin:
    def test_snaps_to_the_nearest_frequency(self):
        assert nearest_bin(FREQS, 740.0) == 2
        assert nearest_bin(FREQS, 10_000.0) == 4
        assert nearest_bin(FREQS, 0.0) == 0

    def test_restricted_to_computed_bins(self):
        """The partial-band case: 1000 Hz is bin 3, but with only bins 0 and 4
        solved the viewer must land on one of those, not colour a NaN."""
        assert nearest_bin(FREQS, 1000.0, allowed=[0, 4]) == 4
        assert nearest_bin(FREQS, 600.0, allowed=[0, 4]) == 0

    def test_a_single_allowed_bin_is_always_the_answer(self):
        assert nearest_bin(FREQS, 1250.0, allowed=[1]) == 1

    def test_no_bins_at_all_raises(self):
        with pytest.raises(ValueError, match="no bins"):
            nearest_bin(FREQS, 500.0, allowed=[])


class _Model:
    """Just enough Model for load_results."""

    def __init__(self, tmp_path):
        self.frequencies = FREQS
        self.fields = {"a": np.zeros((2, 3))}


class TestLoadResults:
    """``results.npz`` is what a finished solve leaves and is far cheaper than
    re-parsing be.out/. It can also be stale: a hand-run NumCalc, or a resume
    still in flight, adds bins underneath it.
    """

    def _npz(self, wd, value):
        wd.mkdir(parents=True, exist_ok=True)
        r = Results(MemoryStore({"pEvalGrid": np.full((2, 5), value,
                                                      dtype=complex)}),
                    FREQS, {"a": np.array([0, 1])})
        return r.save(wd / "results.npz")

    def test_reads_the_npz_when_there_is_no_raw_output(self, tmp_path):
        self._npz(tmp_path, 7.0)
        got = load_results(tmp_path, _Model(tmp_path))
        assert got.pressure("a")[0, 0] == 7.0

    def _raw(self, wd, value):
        """A packed ``be.npz`` — the same thing ``load_bins`` reads out of
        be.out/, minus the ASCII."""
        np.savez(wd / "be.npz",
                 pEvalGrid=np.full((2, 5), value, dtype=np.complex64))

    def test_a_newer_raw_output_wins_over_a_stale_npz(self, tmp_path):
        """The bug this prevents: solve --preview 6, let a resume add bins,
        then open the viewer and see only the first six."""
        import os

        self._npz(tmp_path, 7.0)
        self._raw(tmp_path, 9.0)
        os.utime(tmp_path / "be.npz", (1 << 31, 1 << 31))  # far in the future

        got = load_results(tmp_path, _Model(tmp_path))
        assert got.pressure("a")[0, 0] == 9.0

    def test_the_refreshed_npz_is_written_back(self, tmp_path):
        """``from_beout`` saves, so the next open is cheap again."""
        import os

        self._npz(tmp_path, 7.0)
        self._raw(tmp_path, 9.0)
        os.utime(tmp_path / "be.npz", (1 << 31, 1 << 31))
        load_results(tmp_path, _Model(tmp_path))
        assert Results.load(tmp_path / "results.npz").pressure("a")[0, 0] == 9.0

    def test_an_up_to_date_npz_is_preferred(self, tmp_path):
        """The common case: the npz was written at the end of the solve, so it
        is at least as new as the raw output and must not trigger a re-parse.
        A mutation that always re-ingests would return 9.0 here."""
        self._raw(tmp_path, 9.0)
        self._npz(tmp_path, 7.0)               # written last -> newer
        got = load_results(tmp_path, _Model(tmp_path))
        assert got.pressure("a")[0, 0] == 7.0
