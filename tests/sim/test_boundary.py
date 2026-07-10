"""Tests for bemkit.sim.boundary — geometric BC resolution."""

import meshio
import numpy as np
import pytest

from bemkit.sim.boundary import (
    flatten, resolve_boundary, variant_target,
)


@pytest.fixture()
def mesh_file(tmp_path, tiny_mesh_data):
    vertices, triangles = tiny_mesh_data
    path = tmp_path / "tetra.stl"
    meshio.write_points_cells(path, vertices, [("triangle", triangles)])
    return path, vertices, triangles


def _centroids(vertices, triangles):
    return vertices[triangles].mean(axis=1)


class TestVariantTarget:
    def test_strips_recipe_suffix(self):
        assert variant_target("ear__r05") == "ear"

    def test_strips_profile_suffix(self):
        assert variant_target("ear@bemtype1") == "ear"

    def test_plain_name_unchanged(self):
        assert variant_target("loudspeaker") == "loudspeaker"


class TestResolveBoundary:
    def test_literal_passes_through(self, mesh_file):
        path, _, _ = mesh_file
        cfg = {"BOUNDARY": {"cap": [1, 2, 1.0]}}
        resolved = resolve_boundary(cfg, "tetra", path)
        assert resolved == {"cap": [(1, 2, 1.0)]}

    def test_literal_range_list(self, mesh_file):
        path, _, _ = mesh_file
        cfg = {"BOUNDARY": {"cap": [[0, 1, 1.0], [2, 3, 0.5]]}}
        resolved = resolve_boundary(cfg, "tetra", path)
        assert resolved == {"cap": [(0, 1, 1.0), (2, 3, 0.5)]}

    def test_geometric_single_anchor(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        target = 2
        anchor = cents[target] + 1e-4  # nudge toward element 2's centroid
        cfg = {"BOUNDARY": {"drv": {"near": anchor.tolist()}}}
        resolved = resolve_boundary(cfg, "tetra", path)
        assert resolved["drv"] == [(target, target, 1.0)]

    def test_geometric_multi_anchor(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        cfg = {"BOUNDARY": {"patch": {
            "near": [cents[0].tolist(), cents[3].tolist()], "velocity": 0.5}}}
        resolved = resolve_boundary(cfg, "tetra", path)
        assert resolved["patch"] == [(0, 0, 0.5), (3, 3, 0.5)]

    def test_far_anchor_raises(self, mesh_file):
        path, _, _ = mesh_file
        cfg = {"BOUNDARY": {"drv": {"near": [100.0, 100.0, 100.0]}}}
        with pytest.raises(ValueError, match="from its nearest element"):
            resolve_boundary(cfg, "tetra", path)

    def test_variant_suffix_uses_target_section(self, mesh_file):
        path, _, _ = mesh_file
        # Variant 'ear__r05' must resolve the [BOUNDARY.ear] subsection.
        cfg = {"BOUNDARY": {
            "ear": {"spot": [1, 1, 1.0]},
            "mouth": {"spot": [3, 3, 1.0]},
        }}
        resolved = resolve_boundary(cfg, "ear__r05", path)
        assert resolved == {"spot": [(1, 1, 1.0)]}

    def test_no_boundary_returns_empty(self, mesh_file):
        path, _, _ = mesh_file
        assert resolve_boundary({}, "tetra", path) == {}

    def test_flatten(self):
        assert flatten({"a": [(0, 0, 1.0)], "b": [(1, 2, 0.5), (3, 3, 1.0)]}) == \
            [(0, 0, 1.0), (1, 2, 0.5), (3, 3, 1.0)]
