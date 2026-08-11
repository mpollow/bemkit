"""Tests for SimulationData normalization."""

import warnings
import numpy as np
import pytest

from bemkit.analysis.normalization import NoVelocityDrive, normalize_pressure
from bemkit.analysis.simulation import SimulationData


class TestNormalizePressure:
    """Test the normalize_pressure helper function."""

    def test_normalize_with_A_vib(self):
        """Test normalization applies correct formula."""
        p_raw = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])  # (2, 2)
        frequencies = np.array([100.0, 200.0])
        rho0 = 1.2
        A_vib = 0.1

        result = normalize_pressure(p_raw, frequencies, rho0, A_vib)

        omega = 2 * np.pi * frequencies
        expected = p_raw / (1j * rho0 * omega[None, :] * A_vib)
        np.testing.assert_allclose(result, expected)

    def test_normalize_with_A_vib_none_raises(self):
        """Without a vibrating area the acc normalization has no meaning.

        This used to warn and hand back the *raw* array, so a scattering model
        produced raw Pascals wearing an acceleration-normalized label — wrong
        by a frequency-dependent factor and invisible on a plot. Refusing is
        the only honest answer; `norm="raw"` is the quantity that exists.
        """
        p_raw = np.array([[1 + 1j, 2 + 2j]])  # (1, 2)
        frequencies = np.array([100.0, 200.0])

        with pytest.raises(NoVelocityDrive, match="norm='raw'"):
            normalize_pressure(p_raw, frequencies, 1.2, None)

    def test_normalize_shape_preserved(self):
        """Test output shape matches input."""
        for shape in [(10, 50), (1, 100), (500, 5)]:
            p_raw = np.random.randn(*shape) + 1j * np.random.randn(*shape)
            frequencies = np.linspace(100, 5000, shape[1])
            result = normalize_pressure(p_raw, frequencies, 1.2, 0.05)
            assert result.shape == p_raw.shape


class TestSimulationDataNormalization:
    """Test SimulationData normalized pressure accessors."""

    @pytest.fixture
    def sim(self, numcalc_dir):
        """Create a SimulationData from synthetic numcalc fixture."""
        # Load a minimal config for the synthetic fixture
        config = {
            "sr": 48000,
            "nbins": 65,
            "air_density": 1.2,
            "speed_of_sound": 340.0,
        }
        return SimulationData(numcalc_dir, config)

    def test_frequencies_match_freq_grid(self, sim):
        """The results layer must label frequencies with the same grid the
        runner selects bins from, or a merged partial run would be
        mislabelled."""
        from bemkit.config import freq_grid
        np.testing.assert_array_equal(sim.frequencies, freq_grid(48000, 65))

    def test_pressure_default_is_raw(self, sim):
        """pressure() defaults to norm="raw" — NumCalc's native output is
        the base; acc normalization is opt-in."""
        np.testing.assert_array_equal(
            sim.pressure("all"), sim.pressure("all", norm="raw"))

    def test_pressure_raw_access(self, sim):
        """Test that norm="raw" returns raw data."""
        p_raw = sim.pressure("all", norm="raw")
        p_acc = sim.pressure("all", norm="acc")
        # They should be different (unless A_vib is None)
        if sim.A_vib is not None:
            assert not np.allclose(p_raw, p_acc)
        else:
            np.testing.assert_array_equal(p_raw, p_acc)

    def test_dynamic_attr_p_is_raw(self, sim):
        """sim.p_<set> is the unmarked (raw) form."""
        np.testing.assert_array_equal(sim.p_all,
                                      sim.pressure("all", norm="raw"))

    def test_dynamic_attr_p_acc(self, sim):
        """sim.p_acc_<set> is the normalized opt-in form."""
        np.testing.assert_array_equal(sim.p_acc_all,
                                      sim.pressure("all", norm="acc"))

    def test_velocity_default_is_raw(self, sim):
        """velocity() defaults to norm="raw" (raw 3D vectors)."""
        np.testing.assert_array_equal(
            sim.velocity("all"), sim.velocity("all", norm="raw"))

    def test_velocity_raw_access(self, sim):
        """Test that norm="raw" returns raw 3D velocity."""
        v_raw = sim.velocity("all", norm="raw")
        v_acc = sim.velocity("all", norm="acc")
        # They should be different (unless A_vib is None)
        if sim.A_vib is not None:
            assert not np.allclose(v_raw, v_acc)
        else:
            np.testing.assert_array_equal(v_raw, v_acc)

    def test_dynamic_attr_v_is_raw(self, sim):
        """sim.v_<set> is the unmarked (raw) 3D form."""
        np.testing.assert_array_equal(sim.v_all,
                                      sim.velocity("all", norm="raw"))

    def test_dynamic_attr_v_acc(self, sim):
        """sim.v_acc_<set> is the normalized opt-in 3D form."""
        np.testing.assert_array_equal(sim.v_acc_all,
                                      sim.velocity("all", norm="acc"))

    def test_boundary_pressure_acc(self, sim):
        """Test new boundary_pressure_acc cached property."""
        b_p_acc = sim.boundary_pressure_acc
        assert b_p_acc.dtype == np.complex128
        assert b_p_acc.shape[0] > 0  # has nodes
        assert b_p_acc.shape[1] == len(sim.frequencies)

    def test_result_uses_correct_formula(self, sim):
        """Test that result() applies normalization correctly."""
        result = sim.result("all")
        p_raw = sim.pressure("all", norm="raw")
        p_expected = normalize_pressure(
            p_raw, sim.frequencies, sim._rho0, sim.A_vib
        )
        np.testing.assert_allclose(result.pressure_acc, p_expected)

    def test_result_velocity_acc_is_real_valued(self, sim):
        """Test that result().velocity_acc is real-valued (not imaginary from bug)."""
        result = sim.result("all")
        assert result.velocity_acc.dtype == np.float64, (
            "velocity_acc must be real float, not complex")
        v_raw = sim.velocity("all", norm="raw")
        v_acc_expected = np.linalg.norm(
            normalize_pressure(v_raw, sim.frequencies, sim._rho0, sim.A_vib),
            axis=-1)
        np.testing.assert_allclose(result.velocity_acc, v_acc_expected)

    def test_round_trip_pressure_normalization(self, sim):
        """Dynamic attrs and result() expose the same arrays."""
        result = sim.result("all")
        np.testing.assert_allclose(sim.p_acc_all, result.pressure_acc)
        np.testing.assert_allclose(sim.p_all, result.pressure_raw)

    def test_dir_includes_acc_variants(self, sim):
        """__dir__ advertises the raw and acc spellings, not p_raw_."""
        d = dir(sim)
        for s in sim.set_names:
            assert f"p_{s}" in d
            assert f"p_acc_{s}" in d
            assert f"p_raw_{s}" not in d
