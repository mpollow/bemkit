"""Tests for bemkit.sim.fields — onaxis, polar, plane generators."""

import numpy as np
import pytest

from bemkit.sim.fields import make_fields_onaxis, make_fields_polar, make_fields_plane


# ── make_fields_onaxis ───────────────────────────────────────────────────────

class TestMakeFieldsOnaxis:
    def test_default_x(self):
        pts = make_fields_onaxis(dim='x')
        assert pts.shape == (5, 3)
        np.testing.assert_array_equal(pts[:, 1], 0)
        np.testing.assert_array_equal(pts[:, 2], 0)
        np.testing.assert_array_equal(pts[:, 0], [-1, -0.1, 0, 0.1, 1])

    def test_y_axis(self):
        pts = make_fields_onaxis(dim='y', radii=[0.5, 1.0])
        assert pts.shape == (2, 3)
        np.testing.assert_array_equal(pts[:, 0], 0)
        np.testing.assert_array_equal(pts[:, 1], [0.5, 1.0])
        np.testing.assert_array_equal(pts[:, 2], 0)

    def test_z_axis(self):
        pts = make_fields_onaxis(dim='z', radii=[2.0])
        assert pts.shape == (1, 3)
        assert pts[0, 2] == 2.0
        assert pts[0, 0] == 0.0

    def test_custom_radii(self):
        radii = [0.1, 0.2, 0.3]
        pts = make_fields_onaxis(dim='x', radii=radii)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[:, 0], radii)


# ── make_fields_polar ────────────────────────────────────────────────────────

class TestMakeFieldsPolar:
    def test_default_xy(self):
        pts = make_fields_polar(dims='xy', res=90, radius=1.0)
        assert pts.shape == (4, 3)  # 0, 90, 180, 270
        # At 0°: (1, 0, 0)
        np.testing.assert_allclose(pts[0], [1, 0, 0], atol=1e-15)
        # At 90°: (0, 1, 0)
        np.testing.assert_allclose(pts[1], [0, 1, 0], atol=1e-15)

    def test_xz_plane(self):
        pts = make_fields_polar(dims='xz', res=90, radius=2.0)
        assert pts.shape == (4, 3)
        np.testing.assert_allclose(pts[0], [2, 0, 0], atol=1e-15)
        np.testing.assert_allclose(pts[1], [0, 0, 2], atol=1e-15)

    def test_z_is_zero_for_xy(self):
        pts = make_fields_polar(dims='xy', res=45)
        np.testing.assert_array_equal(pts[:, 2], 0)

    def test_point_count(self):
        pts = make_fields_polar(dims='xy', res=5)
        assert len(pts) == 72  # 360 / 5

    def test_radius(self):
        pts = make_fields_polar(dims='xy', res=30, radius=3.0)
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        np.testing.assert_allclose(r, 3.0)


# ── make_fields_plane ────────────────────────────────────────────────────────

class TestMakeFieldsPlane:
    def test_default_params(self):
        pts = make_fields_plane(dims='xy', res=100, limits=[-200, 200, -200, 200])
        # arange(-200, 201, 100) = [-200, -100, 0, 100, 200] → 5 per axis
        assert pts.shape == (25, 3)
        np.testing.assert_array_equal(pts[:, 2], 0)  # z=0 for xy plane

    def test_units_are_meters(self):
        pts = make_fields_plane(dims='xy', res=100, limits=[-100, 100, -100, 100])
        # limits in mm → points in meters
        assert pts[:, 0].max() == pytest.approx(0.1)
        assert pts[:, 0].min() == pytest.approx(-0.1)

    def test_xz_plane(self):
        pts = make_fields_plane(dims='xz', res=200, limits=[-200, 200, -200, 200])
        np.testing.assert_array_equal(pts[:, 1], 0)  # y=0 for xz plane
        assert pts[:, 0].max() == pytest.approx(0.2)
        assert pts[:, 2].max() == pytest.approx(0.2)

    def test_asymmetric_limits(self):
        pts = make_fields_plane(dims='xy', res=100, limits=[0, 200, -100, 100])
        # x: arange(0, 201, 100) = [0, 100, 200] → 3
        # y: arange(-100, 101, 100) = [-100, 0, 100] → 3
        assert pts.shape == (9, 3)
        assert pts[:, 0].min() == pytest.approx(0.0)
        assert pts[:, 0].max() == pytest.approx(0.2)

    def test_grid_structure(self):
        pts = make_fields_plane(dims='xy', res=50, limits=[-100, 100, -100, 100])
        # 5 x 5 grid
        assert pts.shape == (25, 3)
        unique_x = np.unique(np.round(pts[:, 0], 10))
        unique_y = np.unique(np.round(pts[:, 1], 10))
        assert len(unique_x) == 5
        assert len(unique_y) == 5
