"""Tests for bemkit.meshing.quality — the checks NumCalc does not make.

Each case builds the *defect* explicitly, because a checker that cannot fail is
worth nothing: NumCalc accepts all of these meshes and returns plausible-looking
numbers for every one of them.
"""

import numpy as np
import pytest

from bemkit.meshing.quality import (
    check_element_regularity, check_elements_per_wavelength, check_mesh,
    check_orientation, check_size_uniformity, check_source_clearance,
    check_watertight, worst_level,
)


@pytest.fixture
def octahedron():
    """A closed, consistently outward-wound surface."""
    v = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0],
                  [0, -1, 0], [0, 0, 1.], [0, 0, -1]])
    t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return v, t


class TestWatertight:
    def test_closed_surface_passes(self, octahedron):
        assert check_watertight(*octahedron).level == "ok"

    def test_hole_is_an_error(self, octahedron):
        v, t = octahedron
        f = check_watertight(v, t[:-1])
        assert f.level == "error" and f.detail["boundary_edges"] == 3

    def test_nonmanifold_edge_is_an_error(self, octahedron):
        v, t = octahedron
        f = check_watertight(v, np.vstack([t, t[0:1]]))
        assert f.level == "error" and f.detail["nonmanifold_edges"] == 3


class TestOrientation:
    def test_outward_passes(self, octahedron):
        f = check_orientation(*octahedron)
        assert f.level == "ok" and f.detail["signed_volume"] > 0

    def test_inverted_is_an_error(self, octahedron):
        """The failure mode the paper calls most common: still solves, still
        plots, quietly wrong. NumCalc's own -check_normals passes this."""
        v, t = octahedron
        f = check_orientation(v, t[:, ::-1])
        assert f.level == "error" and "inward" in f.message
        assert f.detail["signed_volume"] < 0

    def test_inconsistent_winding_is_an_error(self, octahedron):
        v, t = octahedron
        t = t.copy()
        t[0] = t[0][::-1]
        f = check_orientation(v, t)
        assert f.level == "error" and "inconsistent" in f.message

    def test_orientation_skipped_on_an_open_mesh(self, octahedron):
        """The signed-volume argument assumes a closed surface, so an open one
        must not get a verdict at all rather than a bogus one."""
        v, t = octahedron
        checks = {f.check for f in check_mesh(v, t[:-1])}
        assert "watertight" in checks and "orientation" not in checks


class TestResolution:
    def test_well_resolved_passes(self, octahedron):
        v, t = octahedron
        # Longest edge ~1.41 m; at 34 Hz the wavelength is 10 m.
        f = check_elements_per_wavelength(v, t, max_frequency=34.0)
        assert f.level == "ok" and f.detail["elements_per_wavelength"] > 6

    def test_marginal_is_a_warning(self, octahedron):
        v, t = octahedron
        f = check_elements_per_wavelength(v, t, max_frequency=60.0)
        assert f.level == "warning"

    def test_badly_under_resolved_is_an_error(self, octahedron):
        v, t = octahedron
        f = check_elements_per_wavelength(v, t, max_frequency=1000.0)
        assert f.level == "error"


class TestElementShape:
    def test_regular_mesh_passes(self, octahedron):
        assert check_element_regularity(*octahedron).level == "ok"

    def test_sliver_is_flagged(self):
        v = np.array([[0., 0, 0], [1, 0, 0], [0.5, 0.001, 0]])
        f = check_element_regularity(v, np.array([[0, 1, 2]]))
        assert f.level == "warning" and f.detail["above_threshold"] == 1

    def test_uniform_sizes_pass(self, octahedron):
        assert check_size_uniformity(*octahedron).level == "ok"

    def test_abrupt_size_jump_is_flagged(self):
        # Two triangles sharing an edge, one 100x the area of the other.
        v = np.array([[0., 0, 0], [1, 0, 0], [0, 0.01, 0], [0, -1.0, 0]])
        t = np.array([[0, 1, 2], [0, 3, 1]])
        f = check_size_uniformity(v, t)
        assert f.level == "warning" and f.detail["above_threshold"] == 1


class TestSourceClearance:
    def test_far_source_passes(self, octahedron):
        v, t = octahedron
        assert check_source_clearance(v, t, [[0, 0, 10.0]]).level == "ok"

    def test_source_on_the_surface_is_flagged(self, octahedron):
        """The free-field Green's function is singular at zero distance, and
        NumCalc reports nothing — it just returns garbage."""
        v, t = octahedron
        f = check_source_clearance(v, t, [[0.33, 0.33, 0.33]])
        assert f.level == "warning" and f.detail["too_close"] == 1

    def test_no_sources_is_fine(self, octahedron):
        assert check_source_clearance(*octahedron, sources=[]).level == "ok"


def test_worst_level_ranks_severity():
    from bemkit.meshing.quality import Finding
    assert worst_level([Finding("a", "ok", ""), Finding("b", "warning", "")]) \
        == "warning"
    assert worst_level([Finding("a", "warning", ""),
                        Finding("b", "error", "")]) == "error"
    assert worst_level([]) == "ok"


def test_check_mesh_runs_everything_applicable(octahedron):
    v, t = octahedron
    checks = {f.check for f in check_mesh(v, t, max_frequency=34.0,
                                          sources=[[0, 0, 10.0]])}
    assert checks == {"watertight", "orientation", "element_regularity",
                      "size_uniformity", "elements_per_wavelength",
                      "source_clearance"}
