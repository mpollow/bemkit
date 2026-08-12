"""The arithmetic behind every colour the 3D viewers paint.

Testable without a display, which is the whole reason it lives apart from
``scene.py``. The two ``add_field_*`` methods used to call a ``_get_values``
helper that had been deleted with the 2D plotting family — an unreachable
NameError sitting in the results path — and nothing here could have caught
that, because nothing here existed.
"""

import numpy as np
import pytest

from bemkit.view._quantities import (
    PHASE_DB_RANGE, impedance_scalars, intensity, magnitude_db,
    quantity_values,
)


class TestQuantityValues:
    Z = np.array([1 + 0j, 0 + 2j, -3 + 0j, 0.5 - 0.5j])

    def test_magnitude(self):
        np.testing.assert_allclose(quantity_values(self.Z, "magnitude"),
                                   np.abs(self.Z))

    def test_magnitude_db(self):
        np.testing.assert_allclose(quantity_values(self.Z, "magnitude_db"),
                                   20 * np.log10(np.abs(self.Z)))

    def test_phase_is_in_degrees_not_radians(self):
        """A silent radian would look like a plausible ±3 range on a plot."""
        got = quantity_values(self.Z, "phase")
        assert got[1] == pytest.approx(90.0)
        assert got[2] == pytest.approx(180.0)

    def test_real_and_imag(self):
        np.testing.assert_allclose(quantity_values(self.Z, "real"), self.Z.real)
        np.testing.assert_allclose(quantity_values(self.Z, "imag"), self.Z.imag)

    def test_a_hard_zero_is_floored_not_minus_infinity(self):
        v = quantity_values(np.array([0j]), "magnitude_db")
        assert np.isfinite(v[0]) and v[0] == pytest.approx(-240.0)

    def test_unknown_quantity_raises_and_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="magnitude_db"):
            quantity_values(self.Z, "loudness")

    def test_shape_is_preserved_for_a_whole_band(self):
        """`_global_range` hands in the full (N, nf) array, not one bin."""
        band = np.ones((4, 7), dtype=complex)
        assert quantity_values(band, "magnitude_db").shape == (4, 7)


class TestPhaseRgba:
    def test_peak_keeps_its_hue_and_the_floor_goes_white(self):
        from bemkit.view._quantities import phase_rgba
        z = np.array([1.0 + 0j, 10 ** (-PHASE_DB_RANGE / 20) * 1j])
        rgba = phase_rgba(z)
        assert rgba.shape == (2, 4)
        assert rgba[:, 3].tolist() == [255, 255]
        # The floor point is white in all three channels; the peak is not.
        assert rgba[1, :3].min() >= 254
        assert rgba[0, :3].min() < 254

    def test_a_shared_reference_shifts_the_fade(self):
        """Passing db_ref is what makes a surface and its eval points fade on
        one scale; without it each actor normalises to its own max and the
        quieter one is drawn as if it were loud."""
        from bemkit.view._quantities import phase_rgba
        z = np.array([1e-3 + 0j, 1e-3 * 1j])
        own = phase_rgba(z)                    # its own max -> full hue
        shared = phase_rgba(z, db_ref=0.0)     # 60 dB down -> faded
        assert shared[0, :3].min() > own[0, :3].min()


class TestImpedanceScalars:
    def test_a_matched_plane_wave_reads_as_one(self):
        """Z0 = p/(v_r rho0 c0) = 1 for an outgoing plane wave. The reading
        the flag exists for: it says the eval grid is far enough out."""
        rho0, c0 = 1.2, 340.0
        pts = np.array([[0., 0, 1], [0, 0, 2]])
        p = np.array([1 + 0j, 1 + 0j])
        v = np.array([[0, 0, 1 / (rho0 * c0)], [0, 0, 1 / (rho0 * c0)]],
                     dtype=complex)
        got = impedance_scalars(pts, p, v, rho0, c0, "real")
        np.testing.assert_allclose(got, [1.0, 1.0], rtol=1e-12)

    def test_only_the_radial_component_counts(self):
        """A purely tangential velocity carries no radial flux; the ratio must
        blow up (NaN), not silently use |v|."""
        pts = np.array([[0., 0, 1]])
        p = np.array([1 + 0j])
        v = np.array([[1.0, 0, 0]], dtype=complex)
        got = impedance_scalars(pts, p, v, 1.2, 340.0, "real")
        assert np.isnan(got[0])

    def test_db_is_of_the_normalised_impedance(self):
        """0 dB is the matched case. The old viewer took dB of the *un*
        normalised Z while labelling the bar Z0, a constant 52 dB offset."""
        rho0, c0 = 1.2, 340.0
        pts = np.array([[0., 0, 1]])
        p = np.array([1 + 0j])
        v = np.array([[0, 0, 1 / (rho0 * c0)]], dtype=complex)
        got = impedance_scalars(pts, p, v, rho0, c0, "magnitude_db")
        assert got[0] == pytest.approx(0.0, abs=1e-9)


class TestIntensity:
    def test_a_plane_wave_carries_p_squared_over_two_rho_c(self):
        rho0, c0 = 1.2, 340.0
        p = np.array([2 + 0j])
        v = np.array([[0, 0, 2 / (rho0 * c0)]], dtype=complex)
        I = intensity(p, v)
        assert I.shape == (1, 3)
        assert I[0, 2] == pytest.approx(4 / (2 * rho0 * c0))

    def test_a_quadrature_field_carries_no_active_intensity(self):
        """p and v 90 degrees apart is a standing/reactive field: Re(p v*) = 0.
        Using |p||v| instead would draw arrows where nothing flows."""
        I = intensity(np.array([1 + 0j]), np.array([[0, 0, 1j]]))
        np.testing.assert_allclose(I, 0.0, atol=1e-15)


def test_magnitude_db_floor_is_configurable():
    assert magnitude_db(np.array([0j]), floor=1.0)[0] == pytest.approx(0.0)
