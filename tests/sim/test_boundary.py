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
        resolved = resolve_boundary({"cap": [1, 2, 1.0]}, "tetra", path)
        assert resolved == {"cap": [(1, 2, "VELO", 1.0, 0.0, None)]}

    def test_literal_range_list(self, mesh_file):
        path, _, _ = mesh_file
        resolved = resolve_boundary(
            {"cap": [[0, 1, 1.0], [2, 3, 0.5]]}, "tetra", path)
        assert resolved == {"cap": [(0, 1, "VELO", 1.0, 0.0, None), (2, 3, "VELO", 0.5, 0.0, None)]}

    def test_geometric_single_anchor(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        target = 2
        anchor = cents[target] + 1e-4  # nudge toward element 2's centroid
        resolved = resolve_boundary(
            {"drv": {"near": anchor.tolist()}}, "tetra", path)
        assert resolved["drv"] == [(target, target, "VELO", 1.0, 0.0, None)]

    def test_geometric_multi_anchor(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        resolved = resolve_boundary(
            {"patch": {"near": [cents[0].tolist(), cents[3].tolist()],
                       "velocity": 0.5}}, "tetra", path)
        assert resolved["patch"] == [(0, 0, "VELO", 0.5, 0.0, None), (3, 3, "VELO", 0.5, 0.0, None)]

    def test_radius_selects_all_elements_within(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        anchor = cents[1]
        r = float(np.linalg.norm(cents - anchor, axis=1).max()) + 1e-6
        resolved = resolve_boundary(
            {"drv": {"near": anchor.tolist(), "radius": r, "velocity": 0.5}},
            "tetra", path)
        assert resolved["drv"] == [(0, 3, "VELO", 0.5, 0.0, None)]  # all 4, collapsed to one range

    def test_radius_selects_subset(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        d = np.sort(np.linalg.norm(cents - cents[1], axis=1))
        r = float(d[1]) / 2  # smaller than the nearest other centroid
        resolved = resolve_boundary(
            {"drv": {"near": cents[1].tolist(), "radius": r}}, "tetra", path)
        assert resolved["drv"] == [(1, 1, "VELO", 1.0, 0.0, None)]

    def test_radius_multi_anchor_union(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        d = np.sort(np.linalg.norm(cents - cents[0], axis=1))
        r = float(d[1]) / 2
        resolved = resolve_boundary(
            {"drv": {"near": [cents[0].tolist(), cents[3].tolist()],
                     "radius": r}}, "tetra", path)
        assert resolved["drv"] == [(0, 0, "VELO", 1.0, 0.0, None), (3, 3, "VELO", 1.0, 0.0, None)]

    def test_radius_with_no_elements_raises(self, mesh_file):
        path, _, _ = mesh_file
        with pytest.raises(ValueError, match="within radius"):
            resolve_boundary({"drv": {"near": [0.5, 0.3, 0.4], "radius": 1e-9}},
                             "tetra", path)

    def test_far_anchor_raises(self, mesh_file):
        path, _, _ = mesh_file
        with pytest.raises(ValueError, match="from its nearest element"):
            resolve_boundary({"drv": {"near": [100.0, 100.0, 100.0]}},
                             "tetra", path)

    def test_variant_suffix_uses_target_section(self, mesh_file):
        path, _, _ = mesh_file
        # Variant 'ear__r05' must resolve the [BOUNDARY.ear] subsection.
        bc = {
            "ear": {"spot": [1, 1, 1.0]},
            "mouth": {"spot": [3, 3, 1.0]},
        }
        resolved = resolve_boundary(bc, "ear__r05", path)
        assert resolved == {"spot": [(1, 1, "VELO", 1.0, 0.0, None)]}

    def test_no_boundary_returns_empty(self, mesh_file):
        path, _, _ = mesh_file
        assert resolve_boundary({}, "tetra", path) == {}
        assert resolve_boundary(None, "tetra", path) == {}

    def test_default_velocity_applies(self, mesh_file):
        path, vertices, triangles = mesh_file
        cents = _centroids(vertices, triangles)
        resolved = resolve_boundary(
            {"drv": {"near": cents[2].tolist()}}, "tetra", path,
            default_velocity=0.25)
        assert resolved["drv"] == [(2, 2, "VELO", 0.25, 0.0, None)]

    def test_flatten(self):
        assert flatten({"a": [(0, 0, "VELO", 1.0, 0.0, None)], "b": [(1, 2, "VELO", 0.5, 0.0, None), (3, 3, "VELO", 1.0, 0.0, None)]}) == \
            [(0, 0, "VELO", 1.0, 0.0, None), (1, 2, "VELO", 0.5, 0.0, None), (3, 3, "VELO", 1.0, 0.0, None)]


class TestLoudFailures:
    """Malformed entries must raise, never silently resolve to 'no BC' —
    a variant without driven elements exports a rigid surface that solves
    to silence."""

    def test_missing_velocity_raises(self):
        with pytest.raises(ValueError, match="cap"):
            resolve_boundary({"cap": [100, 100]}, "v1", mesh_file=None)

    def test_string_entry_raises(self):
        with pytest.raises(ValueError, match="bad"):
            resolve_boundary({"bad": "oops"}, "v1", mesh_file=None)

    def test_unknown_geometric_key_raises(self, mesh_file):
        path, _, _ = mesh_file
        with pytest.raises(ValueError, match="radios"):
            resolve_boundary(
                {"spot": {"near": [0.0, 0.0, 0.05], "radios": 0.01}},
                "v1", mesh_file=path)

    def test_geometric_without_mesh_still_raises_clearly(self):
        with pytest.raises(ValueError, match="near"):
            resolve_boundary({"spot": {"near": [0.0, 0.0, 0.05]}},
                             "v1", mesh_file=None)


class TestBoundaryConditionTypes:
    """PRES / ADMI / IMPE and complex values (Tier 2).

    Element *selection* is unchanged — geometric anchors work exactly as for a
    velocity condition. Only the emitted value differs.
    """

    def test_admittance_with_complex_value(self, mesh_file):
        path, _, _ = mesh_file
        resolved = resolve_boundary(
            {"wall": {"near": _centroids(*mesh_file[1:])[0].tolist(),
                      "radius": 10.0,
                      "type": "admi", "value": [0.4, -0.2]}}, "tetra", path)
        for r in resolved["wall"]:
            assert (r.kind, r.real, r.imag) == ("ADMI", 0.4, -0.2)

    def test_pressure_defaults_to_sound_soft(self, mesh_file):
        path, vertices, triangles = mesh_file
        anchor = _centroids(vertices, triangles)[0].tolist()
        resolved = resolve_boundary(
            {"soft": {"near": anchor, "type": "pres"}}, "tetra", path)
        assert resolved["soft"][0].kind == "PRES"
        assert resolved["soft"][0].real == 0.0     # 0 = sound-soft

    def test_complex_velocity(self, mesh_file):
        """The imaginary part was hardcoded to 0 before Tier 2."""
        path, vertices, triangles = mesh_file
        anchor = _centroids(vertices, triangles)[0].tolist()
        resolved = resolve_boundary(
            {"drv": {"near": anchor, "type": "velo",
                     "value": [1.0, 0.5]}}, "tetra", path)
        assert (resolved["drv"][0].kind, resolved["drv"][0].imag) == ("VELO", 0.5)

    def test_literal_form_is_still_velocity(self, mesh_file):
        path, _, _ = mesh_file
        resolved = resolve_boundary({"cap": [1, 2, 1.0]}, "tetra", path)
        assert resolved["cap"][0].kind == "VELO"

    def test_admi_and_velo_may_share_elements(self, mesh_file):
        """A vibrating source on an absorbing surface: two entries, one range."""
        path, vertices, triangles = mesh_file
        anchor = _centroids(vertices, triangles)[0].tolist()
        resolved = resolve_boundary(
            {"drv": {"near": anchor, "velocity": 1.0},
             "absorb": {"near": anchor, "type": "admi",
                        "value": 0.3}}, "tetra", path)
        kinds = {r.kind for rs in resolved.values() for r in rs}
        assert kinds == {"VELO", "ADMI"}
