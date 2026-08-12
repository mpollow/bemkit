"""Tests for bemkit.model.fields — the frequency grid and resolve_bins.

The ``resolve_bins`` block is rescued verbatim from the deleted
``tests/test_config.py``: the function moved from ``bemkit.config`` to
``bemkit.model.fields`` unchanged, and it is still the thing every
partial-band run goes through, so its behaviour has to stay pinned.
``freq_grid(sr, nbins)`` is now spelled ``fft_grid(sr, nbins)``.
"""

import numpy as np
import pytest

from bemkit.model import fft_grid, resolve_bins


# ── fft_grid ────────────────────────────────────────────────────────────────

class TestFFTGrid:
    def test_basic(self):
        assert len(fft_grid(32000, 65)) == 64

    def test_excludes_dc(self):
        assert fft_grid(32000, 65)[0] == pytest.approx(250.0)

    def test_reaches_nyquist(self):
        assert fft_grid(32000, 65)[-1] == pytest.approx(16000.0)

    def test_spacing_is_uniform(self):
        d = np.diff(fft_grid(48000, 33))
        assert np.allclose(d, d[0])

    def test_single_bin(self):
        assert fft_grid(32000, 2).tolist() == [16000.0]


# ── resolve_bins ────────────────────────────────────────────────────────────

class TestResolveBins:
    """sphcap's grid: sr=32000, nbins=65 → 64 bins, 250 Hz apart."""

    SR, NBINS = 32000, 65

    def _r(self, **kw):
        return resolve_bins(fft_grid(self.SR, self.NBINS), **kw)

    def test_literal_bins_sorted_and_deduped(self):
        assert self._r(bins=[16, 4, 16]) == [4, 16]

    def test_bin_index_above_the_band_raises(self):
        with pytest.raises(ValueError, match="outside the band"):
            self._r(bins=[65])

    def test_bin_index_zero_raises(self):
        """Bins are 1-based because NumCalc's -istart is."""
        with pytest.raises(ValueError, match="1-based"):
            self._r(bins=[0])

    def test_every_out_of_range_bin_is_named(self):
        """The error joins all offenders; fixing them one re-run at a time is
        the thing that convention exists to prevent."""
        with pytest.raises(ValueError, match="0.*99|99.*0"):
            self._r(bins=[0, 4, 99])

    def test_non_integral_bin_raises(self):
        with pytest.raises(ValueError, match="whole numbers"):
            self._r(bins=[3.9])

    def test_bins_accepts_a_numpy_array(self):
        """The freq selector accepts arrays; the two must not disagree about
        what an input may be."""
        assert self._r(bins=np.array([16, 4])) == [4, 16]

    def test_empty_numpy_bins_raises_cleanly(self):
        with pytest.raises(ValueError, match="empty"):
            self._r(bins=np.array([]))

    def test_frequency_snaps_to_nearest_bin(self):
        assert self._r(freq=[1000]) == [4]      # exactly on bin 4
        assert self._r(freq=[1100]) == [4]      # 1100 → 1000 is nearest

    def test_tie_resolves_to_the_lower_bin(self):
        """875 Hz is equidistant from 750 (bin 3) and 1000 (bin 4)."""
        assert self._r(freq=[875]) == [3]

    def test_frequencies_landing_on_one_bin_collapse(self):
        assert self._r(freq=[500, 600]) == [2]

    def test_frequency_above_the_band_raises(self):
        """Silently snapping 30 kHz to the 16 kHz top bin would be a lie."""
        with pytest.raises(ValueError, match="above the simulated band"):
            self._r(freq=[30000])

    def test_frequency_below_the_first_bin_snaps_up(self):
        assert self._r(freq=[10]) == [1]

    def test_nan_frequency_raises(self):
        """NaN passes a > comparison silently and would snap to bin 1."""
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[float("nan")])

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[-500])

    def test_infinite_frequency_raises(self):
        """Before the finiteness guard, inf already raised via the
        above-band check — so this must pin the guard's own message, not
        just any ValueError, or it can't tell the two apart."""
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[float("inf")])

    def test_preview_is_log_spaced(self):
        bins = self._r(preview=8)
        assert bins == [1, 2, 3, 6, 11, 20, 35, 64]
        # log spacing on a linear grid ⇒ non-decreasing gaps between bins
        assert all(np.diff(np.diff(bins)) >= 0)

    def test_preview_deduplicates_at_the_low_end(self):
        """Log spacing crowds the bottom of a linear grid, so a big N yields
        fewer bins than asked."""
        bins = self._r(preview=20)
        assert len(bins) < 20
        assert bins == sorted(set(bins))

    def test_preview_covers_both_ends_of_the_band(self):
        assert self._r(preview=8)[0] == 1
        assert self._r(preview=8)[-1] == 64

    def test_preview_of_one_is_the_lowest_bin(self):
        assert self._r(preview=1) == [1]

    def test_preview_below_one_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            self._r(preview=0)

    def test_two_selectors_raise(self):
        with pytest.raises(ValueError, match="exactly one"):
            self._r(freq=[1000], preview=4)

    def test_no_selector_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            self._r()

    def test_empty_selector_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self._r(freq=[])

    def test_zero_bin_model_raises_clearly(self):
        """nbins=1 is DC only, so the grid is empty; numpy's IndexError names
        nothing useful."""
        with pytest.raises(ValueError, match="no frequency bins"):
            resolve_bins(fft_grid(32000, 1), preview=4)

    def test_every_selector_returns_bins_inside_the_band(self):
        """The merge contract: a partial run's bins must be a subset of the
        full run's, or the later run could not fill in around them."""
        nf = self.NBINS - 1
        for kwargs in ({"preview": 8}, {"freq": [500, 4000]}, {"bins": [1, 64]}):
            assert set(self._r(**kwargs)) <= set(range(1, nf + 1))

    def test_a_non_uniform_band_selects_the_same_way(self):
        """resolve_bins takes the frequency array as data, so a log band (no
        sr/nbins anywhere) snaps and previews identically."""
        band = np.geomspace(100.0, 16000.0, 30)
        assert resolve_bins(band, bins=[30]) == [30]
        assert resolve_bins(band, freq=[100.0]) == [1]
        assert resolve_bins(band, preview=4)[-1] == 30


def test_infinite_bin_index_is_rejected_as_non_integral():
    """floor(inf) == inf, so inf passes an integrality test and then casts to
    INT_MIN — the range error would name -9223372036854775808."""
    with pytest.raises(ValueError, match="whole numbers"):
        resolve_bins(fft_grid(32000, 65), bins=[float("inf")])
