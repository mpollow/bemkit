"""Tests for pynumcalc.packing — be.out -> be.npz."""

import json
import shutil

import numpy as np
import pytest

from pynumcalc.packing import (
    clear_pack,
    has_results,
    is_packed,
    pack_variant,
    unpack_variant,
)
from bemkit.analysis.simulation import SimulationData


@pytest.fixture()
def packed_dir(numcalc_dir):
    """Pack the fixture's be.out/ and return the variant directory."""
    pack_variant(numcalc_dir, verbose=False)
    return numcalc_dir


class TestPackVariant:
    def test_creates_npz(self, numcalc_dir):
        npz_path = pack_variant(numcalc_dir, verbose=False)
        assert npz_path == numcalc_dir / "be.npz"
        assert npz_path.exists()
        assert is_packed(numcalc_dir)

    def test_keeps_be_out_by_default(self, numcalc_dir):
        pack_variant(numcalc_dir, verbose=False)
        assert (numcalc_dir / "be.out").is_dir()

    def test_delete_original_removes_be_out(self, numcalc_dir):
        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        assert not (numcalc_dir / "be.out").exists()
        assert (numcalc_dir / "be.npz").exists()

    def test_archive_contents(self, numcalc_dir):
        pack_variant(numcalc_dir, verbose=False)
        with np.load(numcalc_dir / "be.npz") as nz:
            for k in ("pBoundary", "vBoundary", "pEvalGrid", "vEvalGrid", "bin_indices"):
                assert k in nz.files
            assert nz["pEvalGrid"].dtype == np.complex64
            assert nz["vEvalGrid"].dtype == np.complex64
            assert nz["pEvalGrid"].shape == (3, 2)
            assert nz["vEvalGrid"].shape == (3, 2, 3)
            assert nz["pBoundary"].shape == (4, 2)
            assert nz["vBoundary"].shape == (4, 2)
            np.testing.assert_array_equal(nz["bin_indices"], [1, 2])

    def test_half_written_bin_not_claimed(self, numcalc_dir):
        # A bin killed mid-write (complete pBoundary, empty pEvalGrid) is
        # packed as NaN eval columns but must stay out of bin_indices, so a
        # later resume still sees it as missing and recomputes it.
        (numcalc_dir / "be.out" / "be.2" / "pEvalGrid").write_text("")
        (numcalc_dir / "be.out" / "be.2" / "vEvalGrid").write_text("")
        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        with np.load(numcalc_dir / "be.npz") as nz:
            np.testing.assert_array_equal(nz["bin_indices"], [1])
            assert np.all(np.isnan(nz["pEvalGrid"][:, 1]))
            assert not np.any(np.isnan(nz["pBoundary"][:, 1]))
        from pynumcalc.status import missing_bins
        assert 2 in missing_bins(numcalc_dir, nf=2)

    def test_round_trip_values(self, numcalc_dir):
        """Values written to be.npz match raw ASCII reads within complex64 precision."""
        from pynumcalc.readers import _read_numcalc_data, _read_numcalc_velocity_data

        ascii_pE_b1 = _read_numcalc_data(numcalc_dir / "be.out" / "be.1" / "pEvalGrid")
        ascii_vE_b2 = _read_numcalc_velocity_data(numcalc_dir / "be.out" / "be.2" / "vEvalGrid")

        pack_variant(numcalc_dir, verbose=False)
        with np.load(numcalc_dir / "be.npz") as nz:
            np.testing.assert_allclose(
                nz["pEvalGrid"][:, 0].astype(np.complex128), ascii_pE_b1, rtol=1e-5)
            np.testing.assert_allclose(
                nz["vEvalGrid"][:, 1, :].astype(np.complex128), ascii_vE_b2, rtol=1e-5)

    def test_raises_when_no_be_out(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pack_variant(tmp_path, verbose=False)

    def _stray_bin(self, numcalc_dir, index=999):
        """Leave a truncated be.{index}/pBoundary behind, as an aborted NumCalc does."""
        d = numcalc_dir / "be.out" / f"be.{index}"
        d.mkdir()
        rows = (numcalc_dir / "be.out" / "be.1" / "pBoundary").read_text().splitlines()
        # header claims 4 rows (see conftest), body stops after one
        (d / "pBoundary").write_text("\n".join(rows[:4]) + "\n")
        return d

    def test_stray_bin_does_not_widen_arrays(self, numcalc_dir):
        """An out-of-range be.{i}/ must not set the packed array width."""
        self._stray_bin(numcalc_dir)
        pack_variant(numcalc_dir, nf=2, verbose=False)
        with np.load(numcalc_dir / "be.npz") as nz:
            assert nz["pBoundary"].shape == (4, 2)
            np.testing.assert_array_equal(nz["bin_indices"], [1, 2])

    def test_stray_bin_does_not_block_delete_original(self, numcalc_dir):
        """A truncated stray bin must not abort the pack and strand be.out/."""
        self._stray_bin(numcalc_dir)
        pack_variant(numcalc_dir, nf=2, delete_original=True, verbose=False)
        assert not (numcalc_dir / "be.out").exists()

    def test_truncated_in_range_bin_does_not_block_delete_original(self, numcalc_dir):
        """Same tolerance for a truncated bin with a legitimate index."""
        (numcalc_dir / "be.out" / "be.2" / "pBoundary").write_text(
            "Mesh2HRTF 1.0.0\n    1\n    1 4\n     0  1.0E+00  0.0E+00\n")
        pack_variant(numcalc_dir, nf=2, delete_original=True, verbose=False)
        assert not (numcalc_dir / "be.out").exists()
        with np.load(numcalc_dir / "be.npz") as nz:
            assert np.all(np.isnan(nz["pBoundary"][:, 1]))
            assert 2 not in nz["bin_indices"]

    def test_verify_round_trip_catches_corruption(self, numcalc_dir):
        """The pre-delete gate must reject a be.out/ that diverges from be.npz."""
        from pynumcalc.packing import _verify_round_trip, _write_ascii_quantity
        from pynumcalc.readers import _read_numcalc_data, list_bins

        pack_variant(numcalc_dir, verbose=False)  # keeps be.out/
        be_out = numcalc_dir / "be.out"
        bins = list_bins(be_out)

        # Tamper one ASCII slice so it no longer matches the packed array.
        f = be_out / "be.1" / "pEvalGrid"
        _write_ascii_quantity(f, _read_numcalc_data(f) * 2.0, vector=False)

        with pytest.raises(RuntimeError, match="Round-trip"):
            _verify_round_trip(be_out, numcalc_dir / "be.npz", bins,
                               np.complex64, verbose=False)


class TestSimulationDataPacked:
    """SimulationData should produce identical results in ASCII vs packed mode."""

    def _config(self):
        return {"sr": 48000, "nbins": 3, "air_density": 1.2, "speed_of_sound": 340.0}

    def test_available_bins_match(self, numcalc_dir):
        ascii_sim = SimulationData(numcalc_dir, self._config())
        pack_variant(numcalc_dir, verbose=False)
        packed_sim = SimulationData(numcalc_dir, self._config())
        assert ascii_sim.available_bins == packed_sim.available_bins == [1, 2]

    def test_pressure_parity(self, numcalc_dir):
        cfg = self._config()
        ascii_sim = SimulationData(numcalc_dir, cfg)
        p_ascii = ascii_sim.pressure("all", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        packed_sim = SimulationData(numcalc_dir, cfg)
        p_packed = packed_sim.pressure("all", norm="raw")

        np.testing.assert_allclose(p_ascii, p_packed, rtol=1e-5)
        assert p_packed.dtype == np.complex128

    def test_velocity_parity(self, numcalc_dir):
        cfg = self._config()
        ascii_sim = SimulationData(numcalc_dir, cfg)
        v_ascii = ascii_sim.velocity("all", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        packed_sim = SimulationData(numcalc_dir, cfg)
        v_packed = packed_sim.velocity("all", norm="raw")

        np.testing.assert_allclose(v_ascii, v_packed, rtol=1e-5)
        assert v_packed.shape == v_ascii.shape

    def test_pressure_at_packed(self, numcalc_dir):
        cfg = self._config()
        ascii_sim = SimulationData(numcalc_dir, cfg)
        p1_ascii = ascii_sim.pressure_at(1, domain="boundary", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        packed_sim = SimulationData(numcalc_dir, cfg)
        p1_packed = packed_sim.pressure_at(1, domain="boundary", norm="raw")
        np.testing.assert_allclose(p1_ascii, p1_packed, rtol=1e-5)

    def test_velocity_at_packed(self, numcalc_dir):
        cfg = self._config()
        ascii_sim = SimulationData(numcalc_dir, cfg)
        v1_ascii = ascii_sim.velocity_at(2, domain="eval", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        packed_sim = SimulationData(numcalc_dir, cfg)
        v1_packed = packed_sim.velocity_at(2, domain="eval", norm="raw")
        np.testing.assert_allclose(v1_ascii, v1_packed, rtol=1e-5)
        assert v1_packed.shape == (3, 3)

    def test_boundary_pressure_acc_dtype_preserved(self, packed_dir):
        sim = SimulationData(packed_dir, self._config())
        b_p = sim.boundary_pressure_acc
        assert b_p.dtype == np.complex128


class TestHasResults:
    def test_be_out_only(self, numcalc_dir):
        assert has_results(numcalc_dir)

    def test_packed_only(self, numcalc_dir):
        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        assert has_results(numcalc_dir)

    def test_neither(self, tmp_path):
        assert not has_results(tmp_path)


class TestClearPack:
    def test_no_npz_is_noop(self, numcalc_dir):
        assert clear_pack(numcalc_dir, verbose=False) is False
        assert (numcalc_dir / "be.out").is_dir()

    def test_moves_npz_to_bak(self, numcalc_dir):
        pack_variant(numcalc_dir, verbose=False)
        assert clear_pack(numcalc_dir, verbose=False) is True
        # Moved aside, not deleted — an interrupted re-run can recover it.
        assert not (numcalc_dir / "be.npz").exists()
        assert (numcalc_dir / "be.npz.bak").exists()

    def test_fresh_pack_removes_stale_bak(self, numcalc_dir):
        pack_variant(numcalc_dir, verbose=False)
        clear_pack(numcalc_dir, verbose=False)          # be.npz -> be.npz.bak
        assert (numcalc_dir / "be.npz.bak").exists()
        pack_variant(numcalc_dir, verbose=False)        # fresh pack
        assert (numcalc_dir / "be.npz").exists()
        assert not (numcalc_dir / "be.npz.bak").exists()  # stale backup cleaned

    def test_restore_recreates_be_out(self, numcalc_dir):
        cfg = {"sr": 48000, "nbins": 3, "air_density": 1.2, "speed_of_sound": 340.0}
        original = SimulationData(numcalc_dir, cfg).pressure("all", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        assert clear_pack(numcalc_dir, restore=True, verbose=False) is True
        assert not (numcalc_dir / "be.npz").exists()

        recovered = SimulationData(numcalc_dir, cfg).pressure("all", norm="raw")
        np.testing.assert_allclose(original, recovered, rtol=1e-5)


class TestUnpack:
    def test_round_trip_through_unpack(self, numcalc_dir):
        cfg = {"sr": 48000, "nbins": 3, "air_density": 1.2, "speed_of_sound": 340.0}
        original = SimulationData(numcalc_dir, cfg).pressure("all", norm="raw").copy()

        pack_variant(numcalc_dir, delete_original=True, verbose=False)
        unpack_variant(numcalc_dir, verbose=False)
        assert (numcalc_dir / "be.out").is_dir()

        # be.npz removed for the test? No — unpack leaves it in place.
        # Force ASCII read by deleting npz.
        (numcalc_dir / "be.npz").unlink()
        recovered = SimulationData(numcalc_dir, cfg).pressure("all", norm="raw")
        np.testing.assert_allclose(original, recovered, rtol=1e-5)
