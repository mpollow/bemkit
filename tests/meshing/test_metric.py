"""Smoke tests for the mmgs metric generator."""

import numpy as np

from pynumcalc.meshing.metric import compute_curvature_sizes


def test_curvature_sizes_within_bounds(tiny_mesh_data):
    vertices, triangles = tiny_mesh_data
    sizes, curv_min, curv_max = compute_curvature_sizes(vertices, triangles, hmin=0.001, hmax=0.01)
    assert sizes.shape == (len(vertices),)
    # 1e-12 tolerance: curvature normalization can round a hair below hmin
    assert np.all(sizes >= 0.001 - 1e-12) and np.all(sizes <= 0.01 + 1e-12)
