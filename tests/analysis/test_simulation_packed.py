"""SimulationData access to packed be.npz results.

Single-bin accessors must read from a per-quantity cache — an NpzFile
decompresses the whole member on every __getitem__, so uncached access
would re-decompress the full array per slider tick, and a held-open
NpzFile would keep a stale handle across re-runs.
"""

import numpy as np
import pytest

from bemkit.analysis.simulation import SimulationData
from bemkit.numcalc.packing import pack_variant

CONFIG = {"sr": 8000, "nbins": 3, "air_density": 1.2, "speed_of_sound": 340.0}


@pytest.fixture()
def packed_dir(numcalc_dir):
    pack_variant(numcalc_dir, delete_original=True, verbose=False)
    return numcalc_dir


class TestPackedAccess:
    def test_pressure_at_matches_ascii(self, numcalc_dir):
        p_ascii = SimulationData(numcalc_dir, CONFIG).pressure_at(
            2, domain="eval", norm="raw")
        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        p_npz = SimulationData(numcalc_dir, CONFIG).pressure_at(
            2, domain="eval", norm="raw")
        np.testing.assert_allclose(p_npz, p_ascii, rtol=1e-5)

    def test_velocity_at_matches_ascii(self, numcalc_dir):
        v_ascii = SimulationData(numcalc_dir, CONFIG).velocity_at(
            1, domain="eval", norm="raw")
        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        v_npz = SimulationData(numcalc_dir, CONFIG).velocity_at(
            1, domain="eval", norm="raw")
        np.testing.assert_allclose(v_npz, v_ascii, rtol=1e-5)

    def test_single_bin_access_is_cached(self, packed_dir):
        sim = SimulationData(packed_dir, CONFIG)
        sim.pressure_at(1, domain="eval", norm="raw")
        assert "pEvalGrid" in sim._npz_cache
        first = sim._npz_cache["pEvalGrid"]
        sim.pressure_at(2, domain="eval", norm="raw")
        assert sim._npz_cache["pEvalGrid"] is first  # not re-decompressed

    def test_no_open_npz_handle(self, packed_dir):
        # The be.npz must not be held open for the object's lifetime;
        # clear_pack may move it aside between accesses.
        sim = SimulationData(packed_dir, CONFIG)
        assert not hasattr(sim, "_packed_npz")
        sim.pressure_at(1, domain="eval", norm="raw")  # opens and closes

    def test_missing_quantity_raises_filenotfound(self, packed_dir):
        sim = SimulationData(packed_dir, CONFIG)
        with pytest.raises(FileNotFoundError):
            sim._packed("nonexistent")

    def test_full_load_uses_packed_data(self, packed_dir):
        sim = SimulationData(packed_dir, CONFIG)
        p = sim.pressure("all", norm="raw")
        assert p.shape == (3, 2)
        assert p.dtype == np.complex128
