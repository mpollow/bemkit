"""Tests for bemkit.geometry.sizing — the config→builder mapping (pure, no gmsh)."""

import pytest

from bemkit.geometry.sizing import (
    uniform_field, distance_field, edge_field, geodesic_field,
    patch_edge_curves, sizing_from_config,
)

BASE = {"lc_min": 5.0, "lc_max": 25.0, "dist_min": 50.0, "dist_max": 100.0}


class TestBuilderMetadata:
    def test_each_builder_tags_its_kind(self):
        assert uniform_field(5.0).kind == "uniform"
        assert distance_field(5, 25, 50, 100).kind == "euclidean"
        assert geodesic_field(5, 25, 50, 100).kind == "geodesic"
        assert edge_field(1, 5, 2, 8).kind == "edge"

    def test_builders_are_callable(self):
        assert callable(uniform_field(5.0))
        assert callable(distance_field(5, 25, 50, 100))
        assert callable(edge_field(1, 5, 2, 8))

    def test_edge_field_records_its_curve_selection(self):
        assert edge_field(1, 5, 2, 8).params["curves"] is None
        assert edge_field(1, 5, 2, 8, curves=[7]).params["curves"] == [7]


class _FakeGmsh:
    """The three ``gmsh.model`` calls patch_edge_curves makes, and nothing else.

    A cap face (1) meeting a remainder face (2) at the rim (curve 4), plus the
    two artefacts of a surface of revolution that must not be measured from:
    the seam (curve 1) and the degenerate curve at the pole (curve 3).
    """

    ADJACENT = {1: [1], 3: [1], 4: [1, 2]}

    def __init__(self, boundary):
        self.model = self
        self._boundary = boundary

    def getBoundary(self, dim_tags, oriented=False, combined=True):
        return [(1, tag) for tag in self._boundary]

    def getAdjacencies(self, dim, tag):
        return self.ADJACENT[tag], []


class TestPatchEdgeCurves:
    def test_only_curves_shared_with_another_surface_survive(self):
        """A seam and a pole curve each bound their own surface and nothing
        else, so refining towards them would put a fine meridian stripe and a
        fine spot at the pole into a mesh that asked for a fine rim."""
        assert patch_edge_curves(_FakeGmsh([1, 3, 4]), [1]) == [4]

    def test_orientation_signs_are_dropped(self):
        assert patch_edge_curves(_FakeGmsh([-4]), [1]) == [4]

    def test_a_patch_bordering_nothing_yields_no_curves(self):
        assert patch_edge_curves(_FakeGmsh([1, 3]), [1]) == []

    def test_edge_field_says_so_rather_than_meshing_ungraded(self):
        with pytest.raises(RuntimeError, match="no curves to measure from"):
            edge_field(1, 5, 2, 8)(_FakeGmsh([1, 3]), [1])


class TestSizingFromConfig:
    def test_default_is_euclidean(self):
        assert sizing_from_config(BASE).kind == "euclidean"

    def test_geodesic_selected(self):
        assert sizing_from_config({**BASE, "distance": "geodesic"}).kind == "geodesic"

    def test_explicit_euclidean(self):
        assert sizing_from_config({**BASE, "distance": "euclidean"}).kind == "euclidean"

    def test_params_pass_through(self):
        s = sizing_from_config({**BASE, "distance": "geodesic"})
        assert s.params == {"lc_min": 5.0, "lc_max": 25.0,
                            "dist_min": 50.0, "dist_max": 100.0}
