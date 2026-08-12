"""Curvature-based sizing for mmgs remeshing."""

import numpy as np

from bemkit.geometry.mmg import curvature_sizes, write_sol
from bemkit.model import Mesh


def _cube():
    """Flat faces, sharp edges — curvature varies a lot."""
    v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                  [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                  [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return Mesh(v, t)


def _flat_square():
    """A single planar quad (two coplanar triangles) — zero curvature
    everywhere, and equal min/max curvature exercises the degenerate branch
    of the normalisation."""
    v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    t = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh(v, t)


def test_sizes_are_one_per_vertex_and_within_bounds():
    mesh = _cube()
    s = curvature_sizes(mesh, hmin=0.01, hmax=0.1)
    assert s.shape == (mesh.n_nodes,)
    assert np.all(s >= 0.01) and np.all(s <= 0.1)


def test_flat_mesh_gets_hmax_everywhere_no_divide_by_zero():
    mesh = _flat_square()
    s = curvature_sizes(mesh, hmin=0.01, hmax=0.1)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, 0.1)


def test_write_sol_round_trips_the_count(tmp_path):
    mesh = _cube()
    s = curvature_sizes(mesh, hmin=0.01, hmax=0.1)
    p = tmp_path / "m.sol"
    write_sol(p, s)
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert str(mesh.n_nodes) in lines
