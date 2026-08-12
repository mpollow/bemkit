"""The acceleration normalization, as a spectrum for ``Results.scaled_by``.

The reference is the formula the deleted ``bemkit.analysis.normalization``
applied directly to the arrays::

    p_acc = p_raw / (1j * rho0 * omega * A_vib)

Restoring it as a *scale factor* rather than an array transform is the whole
point: ``Results`` is store-backed and lazy, and the BEM is linear, so a
normalization is the same kind of thing as a driver response.
"""

import numpy as np
import pytest

from bemkit.model import BC, Elements, Medium, Mesh, Model, PlaneWave, Velocity
from bemkit.results.normalization import NoVelocityDrive, acceleration_scale
from bemkit.results.results import Results
from bemkit.results.store import MemoryStore

FREQS = np.array([250.0, 500.0, 750.0])

#: Two unit right triangles in z=0, area 1/2 each.
VERTS = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
TRIS = np.array([[0, 1, 2], [1, 3, 2]])


def _model(bcs, rho=1.2):
    return Model(Mesh(VERTS, TRIS), FREQS, Medium(rho=rho), bcs=bcs)


def _reference(raw, frequencies, rho0, a_vib):
    """The deleted normalize_pressure, verbatim, as the oracle."""
    omega = 2 * np.pi * frequencies
    if raw.ndim == 2:
        omega_bc = omega[None, :]
    else:
        omega_bc = omega[None, :, None]
    return raw / (1j * rho0 * omega_bc * a_vib)


def test_scale_is_the_reciprocal_of_j_rho_omega_area():
    m = _model([BC(Elements([0]), Velocity(1.0))], rho=1.3)
    scale = acceleration_scale(m)
    expected = 1.0 / (1j * 1.3 * 2 * np.pi * FREQS * 0.5)
    np.testing.assert_allclose(scale, expected, rtol=1e-12)


def test_no_velocity_drive_is_refused_by_name():
    m = _model([], )
    m.sources = (PlaneWave([0, 0, 1]),)
    with pytest.raises(NoVelocityDrive, match="no velocity-driven"):
        acceleration_scale(m)


def test_zero_velocity_alone_is_no_drive():
    """VELO 0.0 is sound-hard; it must not produce a scale of 1/0."""
    m = _model([BC(Elements([0]), Velocity(0.0))])
    with pytest.raises(NoVelocityDrive):
        acceleration_scale(m)


class TestScaledByReproducesTheOldFormula:
    """The reason to check: ``scaled_by`` broadcasts a (nf,) spectrum against
    both a (N, nf) pressure and a (N, nf, 3) velocity, and getting the axis
    wrong on the 3-vector is silent — it would still be a valid multiply.
    """

    def _results(self):
        rng = np.random.default_rng(0)
        p = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        v = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
        pb = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
        store = MemoryStore({"pEvalGrid": p, "vEvalGrid": v, "pBoundary": pb})
        return Results(store, FREQS, {"a": np.array([0, 1, 2, 3])}), p, v, pb

    def test_pressure_matches(self):
        r, p, _, _ = self._results()
        m = _model([BC(Elements([0]), Velocity(1.0))])
        got = r.scaled_by(acceleration_scale(m)).pressure("a")
        np.testing.assert_allclose(got, _reference(p, FREQS, 1.2, 0.5),
                                   rtol=1e-12)

    def test_velocity_matches_on_the_three_vector(self):
        r, _, v, _ = self._results()
        m = _model([BC(Elements([0]), Velocity(1.0))])
        got = r.scaled_by(acceleration_scale(m)).velocity("a")
        np.testing.assert_allclose(got, _reference(v, FREQS, 1.2, 0.5),
                                   rtol=1e-12)

    def test_the_three_vector_is_scaled_along_frequency_not_component(self):
        """Mutation guard: with nf == 3 and a 3-vector, a scale applied to the
        wrong axis has the right shape and the wrong answer. Feed a spectrum
        that differs per frequency and check one element by hand.
        """
        r, _, v, _ = self._results()
        scale = np.array([1.0, 10.0, 100.0], dtype=complex)
        got = r.scaled_by(scale).velocity("a")
        # point 0, frequency bin 1, component 2 -> scaled by 10, not by 100.
        assert got[0, 1, 2] == pytest.approx(v[0, 1, 2] * 10.0)

    def test_boundary_pressure_matches(self):
        r, _, _, pb = self._results()
        m = _model([BC(Elements([0]), Velocity(1.0))])
        got = r.scaled_by(acceleration_scale(m)).boundary_pressure()
        np.testing.assert_allclose(got, _reference(pb, FREQS, 1.2, 0.5),
                                   rtol=1e-12)
