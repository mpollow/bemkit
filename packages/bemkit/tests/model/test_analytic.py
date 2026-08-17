"""Tests for bemkit.model.analytic — the exact spherical-cap solution.

This is what a solver gets checked against, so it needs checking itself. The
old ``tests/analysis/test_analytic_sphcap.py`` only compared it to a stored
NumCalc run under ``tests/models/sphcap/`` — a path that does not exist in
this tree, so every one of its tests had always been skipped and the module
was, in practice, untested. What is pinned here instead runs unconditionally
and needs no BEM data: closed forms and invariants the expansion must satisfy.

The key case is ``angle_deg=180``: the cap then covers the whole sphere, only
the n=0 channel survives, and the field collapses to the textbook pulsating
sphere ``p(r) = -j·rho0·c0·h0(kr) / h0'(ka)`` — which also fixes the
normalization convention (a unit *velocity* boundary condition) end to end.
"""

import re

import numpy as np
import pytest

from bemkit.model.analytic import sphcap_pressure

scipy = pytest.importorskip("scipy.special")

RADIUS_MM = 50.0
RADIUS_M = RADIUS_MM / 1e3
C0 = 340.0
RHO0 = 1.2
FREQS = np.array([250.0, 1000.0, 4000.0])


def _hn2(n, z, derivative=False):
    """Spherical Hankel of the second kind, computed independently of the
    module under test."""
    return (scipy.spherical_jn(n, z, derivative)
            - 1j * scipy.spherical_yn(n, z, derivative))


def _p(points, *, angle_deg=180.0, nmax=12, freqs=FREQS):
    return sphcap_pressure(np.asarray(points, dtype=float), freqs,
                           c0=C0, rho0=RHO0, angle_deg=angle_deg,
                           radius_mm=RADIUS_MM, nmax=nmax)


class TestFullCapIsAPulsatingSphere:
    """angle_deg=180 leaves only n=0, whose closed form is known exactly."""

    POINTS = np.array([[0, 0, 0.3], [0.3, 0, 0], [0, 0, 1.0], [0, 0, -0.6]])

    def test_matches_the_monopole_closed_form(self):
        got = _p(self.POINTS)
        r = np.linalg.norm(self.POINTS, axis=1)
        k = 2 * np.pi * FREQS / C0
        want = (-1j * RHO0 * C0
                * _hn2(0, k[None, :] * r[:, None])
                / _hn2(0, k * RADIUS_M, True)[None, :])
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_field_is_spherically_symmetric(self):
        """Same radius, six directions — a monopole cannot tell them apart."""
        r = 0.4
        dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                         [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float) * r
        p = _p(dirs)
        np.testing.assert_allclose(p, np.tile(p[0], (len(dirs), 1)), rtol=1e-10)

    def test_magnitude_decays_as_one_over_r(self):
        radii = np.array([0.25, 0.5, 1.0, 2.0])
        p = _p(np.column_stack([np.zeros(4), np.zeros(4), radii]))
        scaled = np.abs(p) * radii[:, None]
        np.testing.assert_allclose(scaled, np.tile(scaled[0], (len(radii), 1)),
                                   rtol=1e-10)

    def test_truncation_order_does_not_matter(self):
        """Only n=0 is excited, so raising nmax must change nothing."""
        np.testing.assert_allclose(_p(self.POINTS, nmax=4),
                                   _p(self.POINTS, nmax=40), rtol=1e-10)


def _radial_velocity(points, *, angle_deg, nmax, freq=1000.0, h=1e-7):
    """v_r from Euler's equation, by differencing p across a radial step."""
    points = np.asarray(points, dtype=float)
    unit = points / np.linalg.norm(points, axis=1, keepdims=True)
    f = np.array([freq])
    inner = _p(unit * (RADIUS_M + h), angle_deg=angle_deg, nmax=nmax, freqs=f)
    outer = _p(unit * (RADIUS_M + 2 * h), angle_deg=angle_deg, nmax=nmax, freqs=f)
    return -((outer - inner) / h)[:, 0] / (1j * RHO0 * 2 * np.pi * freq)


def test_full_cap_has_unit_surface_velocity():
    """The docstring's contract: the solution is for a unit *velocity* BC.

    A sign or rho0/omega slip anywhere in the normalization chain shows up
    here as a velocity that is not 1.
    """
    dirs = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0]], dtype=float)
    v = _radial_velocity(dirs, angle_deg=180.0, nmax=12)
    np.testing.assert_allclose(v, 1.0, atol=1e-3)


def test_a_partial_cap_drives_only_the_cap():
    """Inside a 60° cap the surface moves; the far side is nearly rigid.

    Only approximately — a truncated SH series rings at the cap edge — but a
    cap pointing the wrong way, or applied to the whole sphere, fails wide.
    """
    v = _radial_velocity(np.array([[0, 0, 1], [0, 0, -1]], dtype=float),
                         angle_deg=60.0, nmax=60)
    assert abs(v[0]) == pytest.approx(1.0, abs=0.15)
    assert abs(v[1]) < 0.05


def test_zero_frequency_is_left_at_zero():
    """f=0 is skipped rather than dividing by k=0."""
    p = _p([[0, 0, 0.3]], freqs=np.array([0.0, 1000.0]))
    assert p[0, 0] == 0
    assert p[0, 1] != 0


def test_shape_and_dtype():
    p = _p([[0, 0, 0.3], [0, 0, 0.5]])
    assert p.shape == (2, len(FREQS))
    assert np.iscomplexobj(p)


def test_a_partial_cap_is_louder_on_axis_than_behind():
    """A 20° cap is a baffled patch: the front lobe dominates the back."""
    p = _p([[0, 0, 1.0], [0, 0, -1.0]], angle_deg=20.0, nmax=40)
    assert np.all(np.abs(p[0]) > np.abs(p[1]))


class TestTruncationOrder:
    """``nmax`` has a ceiling as well as a floor, which is not obvious.

    ``y_n(x) ~ -(2n-1)!!/x^(n+1)`` overflows to infinity for n well past x, and
    ``1j * inf`` is ``nan + inf j`` — so asking for a *higher* order does not
    buy accuracy past that point, it returns NaN. It used to do so silently.
    """

    POINTS = np.array([[0, 0, 0.3], [0.3, 0, 0], [0, 0, 1.0]])
    BAND = np.array([250.0, 4000.0, 16000.0])

    def test_the_default_order_is_converged(self):
        """Chosen from k*r0, and it has to agree with a hand-picked order that
        is known to work — 80 is what tests/solvers/test_cross_validation.py
        passes."""
        auto = _p(self.POINTS, angle_deg=30.0, nmax=None, freqs=self.BAND)
        pinned = _p(self.POINTS, angle_deg=30.0, nmax=80, freqs=self.BAND)
        np.testing.assert_allclose(auto, pinned, rtol=1e-9)

    def test_the_default_order_is_finite(self):
        p = _p(self.POINTS, angle_deg=30.0, nmax=None, freqs=self.BAND)
        assert np.all(np.isfinite(p))

    def test_an_order_that_would_return_nan_raises_instead(self):
        with pytest.raises(ValueError, match="not representable"):
            _p(self.POINTS, angle_deg=30.0, nmax=200, freqs=self.BAND)

    def test_the_refusal_names_an_order_that_works(self):
        """An error that only says no is a worse reference than one that says
        what to pass instead — so the order it suggests has to be usable."""
        with pytest.raises(ValueError) as excinfo:
            _p(self.POINTS, angle_deg=30.0, nmax=200, freqs=self.BAND)
        suggested = int(re.search(r"have (\d+) chosen", str(excinfo.value))[1])
        assert np.all(np.isfinite(
            _p(self.POINTS, angle_deg=30.0, nmax=suggested, freqs=self.BAND)))
