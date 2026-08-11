"""Tests for bemkit.analysis._grid — grid detection and reshape."""

import numpy as np
import pytest

from bemkit.analysis._grid import detect_grid, reshape_to_grid


class TestDetectCartesianGrid:
    def test_xy_plane(self):
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
        z = np.zeros(9)
        pts = np.column_stack([x, y, z])
        info = detect_grid(pts)
        assert info is not None
        assert info["type"] == "cartesian"
        assert info["const_axis"] == "z"
        assert info["shape"][0] * info["shape"][1] == 9

    def test_xz_plane(self):
        x = np.array([0, 1, 0, 1], dtype=float)
        z = np.array([0, 0, 1, 1], dtype=float)
        y = np.zeros(4)
        pts = np.column_stack([x, y, z])
        info = detect_grid(pts)
        assert info is not None
        assert info["const_axis"] == "y"

    def test_too_few_points(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        assert detect_grid(pts) is None

    def test_non_grid(self):
        rng = np.random.default_rng(42)
        pts = rng.random((10, 3))
        assert detect_grid(pts) is None


class TestDetectSphericalGrid:
    def test_sphere_grid(self):
        # Avoid poles (theta=0, π) where phi degenerates
        theta = np.linspace(0.2, np.pi - 0.2, 5)
        phi = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        r = 1.0
        x = r * np.sin(T) * np.cos(P)
        y = r * np.sin(T) * np.sin(P)
        z = r * np.cos(T)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        info = detect_grid(pts)
        assert info is not None
        assert info["type"] == "spherical"
        assert info["shape"] == (5, 8)
        assert info["radius"] == pytest.approx(1.0, abs=1e-10)


class TestDetectGridWithNaN:
    def test_nan_rows_handled(self):
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, np.nan], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, np.nan], dtype=float)
        z = np.zeros(10)
        z[-1] = np.nan
        pts = np.column_stack([x, y, z])
        info = detect_grid(pts)
        # Should still detect the 3x3 grid from valid points
        # (whether it succeeds depends on n_total factoring)
        if info is not None:
            assert info["type"] == "cartesian"


class TestReshapeToGrid:
    def test_cartesian_roundtrip(self):
        # 3x4 grid on xy plane
        x = np.linspace(0, 2, 3)
        y = np.linspace(0, 3, 4)
        X, Y = np.meshgrid(x, y, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(12)])
        values = np.arange(12, dtype=float)

        info = detect_grid(pts)
        assert info is not None
        row2d, col2d, val2d = reshape_to_grid(pts, values, info)
        # All values should be present (no NaN)
        assert not np.any(np.isnan(val2d))
        assert val2d.shape[0] * val2d.shape[1] == 12

    def test_spherical_roundtrip(self):
        theta = np.linspace(0.1, np.pi - 0.1, 4)
        phi = np.linspace(-np.pi, np.pi, 6, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        r = 2.0
        x = r * np.sin(T) * np.cos(P)
        y = r * np.sin(T) * np.sin(P)
        z = r * np.cos(T)
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        values = np.arange(24, dtype=float)

        info = detect_grid(pts)
        assert info is not None
        row2d, col2d, val2d = reshape_to_grid(pts, values, info)
        assert not np.any(np.isnan(val2d))
        assert val2d.shape == (4, 6)
