"""Tests for bemkit.geometry.geodesic — over-the-surface distance and size ramp."""

import numpy as np
import pytest

from bemkit.geometry.geodesic import (
    geodesic_distance,
    size_from_distance,
    pack_scalar_triangles,
)


def _structured_grid(P):
    """Triangulate a structured (nu, nv, 3) point grid into (nodes, triangles).

    Quads are split along the a-c diagonal, giving edge directions x, y and one
    diagonal — the usual triangulated-grid connectivity.
    """
    nu, nv = P.shape[:2]
    nodes = P.reshape(-1, 3)
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    triangles = np.vstack([np.column_stack([a, b, c]),
                           np.column_stack([a, c, d])])
    return nodes, triangles


def _folded_strip(length=1.0, gap=0.02, width=0.2, n_arm=41, n_width=5):
    """A strip folded back on itself — the horn geometry in miniature.

    Path in the x-z plane: out along z=0 to x=length, up by `gap`, back to x=0
    at z=gap. The two free ends sit `gap` apart in space but a full 2*length
    apart over the surface. Extruded in y to make a surface.

    Returns (nodes, triangles, start_nodes, end_nodes).
    """
    out = np.column_stack([np.linspace(0, length, n_arm), np.zeros(n_arm)])
    fold = np.column_stack([np.full(3, length), np.linspace(0, gap, 3)])[1:]
    back = np.column_stack([np.linspace(length, 0, n_arm), np.full(n_arm, gap)])[1:]
    path = np.vstack([out, fold, back])                      # (n_path, 2) as (x, z)

    y = np.linspace(0, width, n_width)
    P = np.empty((len(path), n_width, 3))
    P[:, :, 0] = path[:, 0, None]
    P[:, :, 1] = y[None, :]
    P[:, :, 2] = path[:, 1, None]

    nodes, triangles = _structured_grid(P)
    idx = np.arange(len(path) * n_width).reshape(len(path), n_width)
    return nodes, triangles, idx[0], idx[-1]


class TestFoldedStrip:
    """The regression test that defines the feature.

    Under a Euclidean metric the far arm of a fold looks adjacent to the source
    and gets needlessly refined; under a geodesic metric it correctly falls to
    lc_max. If this passes and the others fail, the feature still works; if this
    fails, it has silently regressed to Euclidean behaviour.
    """

    def test_far_arm_is_euclidean_near_but_geodesically_far(self):
        L, gap = 1.0, 0.02
        nodes, tris, start, end = _folded_strip(length=L, gap=gap)

        euclid = np.linalg.norm(nodes[start[0]] - nodes[end[0]])
        assert euclid < 2 * gap, "ends should be adjacent in space"

        d = geodesic_distance(nodes, tris, start)
        assert d[end].min() > 1.5 * L, "surface path must run around the fold"

    def test_far_arm_is_not_refined(self):
        L = 1.0
        nodes, tris, start, end = _folded_strip(length=L)
        lc_min, lc_max = 1.0, 10.0
        # Ramp ends well before the geodesic distance to the far arm (~2L).
        ramp = dict(lc_min=lc_min, lc_max=lc_max, dist_min=0.1, dist_max=1.0)

        d = geodesic_distance(nodes, tris, start)
        sizes = size_from_distance(d, **ramp)
        assert np.allclose(sizes[end], lc_max)

        # Contrast: the ambient distance GMSH's Distance field would use — to
        # the nearest point of the whole source region — puts the far arm
        # inside the refined zone. This is the bug being fixed.
        euclid = np.linalg.norm(
            nodes[:, None, :] - nodes[None, start, :], axis=2).min(axis=1)
        assert np.allclose(size_from_distance(euclid, **ramp)[end], lc_min)

    def test_source_arm_is_refined(self):
        nodes, tris, start, _ = _folded_strip()
        sizes = size_from_distance(
            geodesic_distance(nodes, tris, start),
            lc_min=1.0, lc_max=10.0, dist_min=0.1, dist_max=1.0)
        assert np.allclose(sizes[start], 1.0)


class TestFlatPlate:
    """On a flat plate the geodesic distance is the Euclidean one, up to the
    metrication error from constraining paths to mesh edges."""

    @staticmethod
    def _plate(n=41):
        g = np.linspace(0.0, 1.0, n)
        P = np.zeros((n, n, 3))
        P[:, :, 0] = g[:, None]
        P[:, :, 1] = g[None, :]
        return _structured_grid(P)

    def test_never_underestimates(self):
        nodes, tris = self._plate()
        d = geodesic_distance(nodes, tris, [0])
        true = np.linalg.norm(nodes - nodes[0], axis=1)
        assert np.all(d >= true - 1e-12)

    def test_metrication_error_is_small(self):
        nodes, tris = self._plate()
        d = geodesic_distance(nodes, tris, [0])
        true = np.linalg.norm(nodes - nodes[0], axis=1)
        far = true > 0.1                        # near field is dominated by one edge
        # Worst case on a grid with one diagonal is ~8.3%, at 22.5 degrees off
        # the axis, where a path must alternate axis and diagonal steps.
        assert np.all(d[far] <= 1.10 * true[far])


class TestSizeFromDistance:
    RAMP = dict(lc_min=2.0, lc_max=12.0, dist_min=50.0, dist_max=100.0)

    def test_ramp_and_clamping(self):
        d = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 1e6, np.inf])
        got = size_from_distance(d, **self.RAMP)
        expected = [2.0, 2.0, 2.0, 7.0, 12.0, 12.0, 12.0]
        assert np.allclose(got, expected)

    def test_unreachable_becomes_lc_max(self):
        got = size_from_distance(np.array([np.inf]), **self.RAMP)
        assert got[0] == self.RAMP["lc_max"]

    def test_degenerate_ramp_is_a_step(self):
        d = np.array([0.0, 50.0, 50.001, np.inf])
        got = size_from_distance(d, lc_min=2.0, lc_max=12.0,
                                 dist_min=50.0, dist_max=50.0)
        assert np.allclose(got, [2.0, 2.0, 12.0, 12.0])

    def test_returns_float_array(self):
        got = size_from_distance([0, 100], **self.RAMP)
        assert got.dtype == np.float64


class TestDisconnected:
    @staticmethod
    def _two_islands():
        nodes = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [9.0, 0.0, 0.0], [10.0, 0.0, 0.0], [9.0, 1.0, 0.0],
        ])
        triangles = np.array([[0, 1, 2], [3, 4, 5]])
        return nodes, triangles

    def test_unreachable_component_is_inf(self):
        nodes, tris = self._two_islands()
        d = geodesic_distance(nodes, tris, [0])
        assert np.all(np.isfinite(d[:3]))
        assert np.all(np.isinf(d[3:]))

    def test_unreachable_component_sized_at_lc_max(self):
        nodes, tris = self._two_islands()
        sizes = size_from_distance(geodesic_distance(nodes, tris, [0]),
                                   lc_min=1.0, lc_max=5.0,
                                   dist_min=0.5, dist_max=2.0)
        assert np.allclose(sizes[3:], 5.0)


class TestSources:
    def test_multi_source_is_elementwise_min(self):
        nodes, tris = TestFlatPlate._plate(n=21)
        both = geodesic_distance(nodes, tris, [0, 200])
        a = geodesic_distance(nodes, tris, [0])
        b = geodesic_distance(nodes, tris, [200])
        assert np.allclose(both, np.minimum(a, b))

    def test_source_nodes_are_zero(self):
        nodes, tris = TestFlatPlate._plate(n=11)
        d = geodesic_distance(nodes, tris, [0, 55])
        assert d[0] == 0.0 and d[55] == 0.0

    def test_empty_source_raises(self):
        nodes, tris = TestFlatPlate._plate(n=11)
        with pytest.raises(ValueError, match="empty source"):
            geodesic_distance(nodes, tris, [])


class TestPackScalarTriangles:
    def test_coordinate_major_layout(self):
        """GMSH wants x1 x2 x3 y1 y2 y3 z1 z2 z3 v1 v2 v3, not vertex-major.

        Hand-written expectation, because a transposed packing produces a field
        GMSH accepts without error — it is only visible in the resulting mesh.
        """
        nodes = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        triangles = np.array([[0, 1, 2]])
        values = np.array([10.0, 11.0, 12.0])

        got = pack_scalar_triangles(nodes, triangles, values)
        expected = [1.0, 4.0, 7.0,      # x's
                    2.0, 5.0, 8.0,      # y's
                    3.0, 6.0, 9.0,      # z's
                    10.0, 11.0, 12.0]   # values
        assert np.allclose(got, expected)

    def test_twelve_doubles_per_triangle(self):
        nodes, triangles = TestFlatPlate._plate(n=11)
        values = np.arange(len(nodes), dtype=float)
        got = pack_scalar_triangles(nodes, triangles, values)
        assert got.shape == (12 * len(triangles),)

    def test_triangles_are_packed_in_order(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        got = pack_scalar_triangles(nodes, triangles, values).reshape(2, 12)
        assert np.allclose(got[0, 9:], [1.0, 2.0, 3.0])
        assert np.allclose(got[1, 9:], [2.0, 4.0, 3.0])
