"""Tests for bemkit.meshing.sizing — the config→builder mapping (pure, no gmsh)."""

import pytest

from bemkit.meshing.sizing import (
    uniform_field, distance_field, geodesic_field, sizing_from_config,
)

BASE = {"lc_min": 5.0, "lc_max": 25.0, "dist_min": 50.0, "dist_max": 100.0}


class TestBuilderMetadata:
    def test_each_builder_tags_its_kind(self):
        assert uniform_field(5.0).kind == "uniform"
        assert distance_field(5, 25, 50, 100).kind == "euclidean"
        assert geodesic_field(5, 25, 50, 100).kind == "geodesic"

    def test_builders_are_callable(self):
        assert callable(uniform_field(5.0))
        assert callable(distance_field(5, 25, 50, 100))


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
