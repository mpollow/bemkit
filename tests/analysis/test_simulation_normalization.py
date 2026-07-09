"""Tests for SimulationData normalization."""

import warnings
import numpy as np
import pytest

from pynumcalc.analysis.normalization import normalize_pressure
from pynumcalc.analysis.simulation import SimulationData


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

    def test_normalize_with_A_vib_none(self):
        """Test that None A_vib returns copy with warning."""
        p_raw = np.array([[1 + 1j, 2 + 2j]])  # (1, 2)
        frequencies = np.array([100.0, 200.0])
        rho0 = 1.2

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_pressure(p_raw, frequencies, rho0, None)
            assert len(w) == 1
            assert "No vibrating elements found" in str(w[0].message)

        np.testing.assert_array_equal(result, p_raw)
        assert result is not p_raw  # should be a copy

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

    def test_pressure_default_is_normalized(self, sim):
        """Test that pressure() defaults to norm="acc"."""
        p_acc = sim.pressure("all")
        p_acc_explicit = sim.pressure("all", norm="acc")
        np.testing.assert_array_equal(p_acc, p_acc_explicit)

    def test_pressure_raw_access(self, sim):
        """Test that norm="raw" returns raw data."""
        p_raw = sim.pressure("all", norm="raw")
        p_acc = sim.pressure("all", norm="acc")
        # They should be different (unless A_vib is None)
        if sim.A_vib is not None:
            assert not np.allclose(p_raw, p_acc)
        else:
            np.testing.assert_array_equal(p_raw, p_acc)

    def test_dynamic_attr_p_is_normalized(self, sim):
        """Test that sim.p_<set> returns normalized."""
        p_via_attr = sim.p_all
        p_via_method = sim.pressure("all", norm="acc")
        np.testing.assert_array_equal(p_via_attr, p_via_method)

    def test_dynamic_attr_p_raw(self, sim):
        """Test that sim.p_raw_<set> returns raw."""
        p_raw_via_attr = sim.p_raw_all
        p_raw_via_method = sim.pressure("all", norm="raw")
        np.testing.assert_array_equal(p_raw_via_attr, p_raw_via_method)

    def test_velocity_default_is_normalized(self, sim):
        """Test that velocity() defaults to norm="acc" (normalized 3D)."""
        v_acc = sim.velocity("all")
        v_acc_explicit = sim.velocity("all", norm="acc")
        np.testing.assert_array_equal(v_acc, v_acc_explicit)

    def test_velocity_raw_access(self, sim):
        """Test that norm="raw" returns raw 3D velocity."""
        v_raw = sim.velocity("all", norm="raw")
        v_acc = sim.velocity("all", norm="acc")
        # They should be different (unless A_vib is None)
        if sim.A_vib is not None:
            assert not np.allclose(v_raw, v_acc)
        else:
            np.testing.assert_array_equal(v_raw, v_acc)

    def test_dynamic_attr_v_is_normalized(self, sim):
        """Test that sim.v_<set> returns normalized 3D velocity."""
        v_via_attr = sim.v_all
        v_via_method = sim.velocity("all", norm="acc")
        np.testing.assert_array_equal(v_via_attr, v_via_method)

    def test_dynamic_attr_v_raw(self, sim):
        """Test that sim.v_raw_<set> returns raw 3D velocity."""
        v_raw_via_attr = sim.v_raw_all
        v_raw_via_method = sim.velocity("all", norm="raw")
        np.testing.assert_array_equal(v_raw_via_attr, v_raw_via_method)

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
        """Test that dynamic attr matches result()."""
        p_via_attr = sim.p_all
        result = sim.result("all")
        np.testing.assert_allclose(p_via_attr, result.pressure_acc)

    def test_dir_includes_p_raw(self, sim):
        """Test that __dir__ includes p_raw_<set> entries."""
        d = dir(sim)
        for s in sim.set_names:
            assert f"p_{s}" in d
            assert f"p_raw_{s}" in d
