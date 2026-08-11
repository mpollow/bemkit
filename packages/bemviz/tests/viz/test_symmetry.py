"""Tests for bemviz.viz._symmetry — ghost geometry for mirrored meshes.

Data preparation only. Rendering is not tested: screenshot comparison of live
pyvista windows has proven unreliable in this repo and would add no
verification value over checking the arrays handed to the scene.
"""

import numpy as np
import pytest

from bemviz.viz._symmetry import ghost_geometry, describe_symmetry


def _two_triangles():
    """Two triangles in the x, y >= 0 quadrant, one per 'physical group'."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    return nodes, elements


class TestGhostGeometryIsOptional:
    def test_returns_none_without_symmetry(self):
        nodes, elements = _two_triangles()
        assert ghost_geometry(nodes, elements, None) is None

    def test_returns_none_when_every_plane_is_inactive(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [0, 0, 0], "coordinates": [0.0, 0.0, 0.0]}
        assert ghost_geometry(nodes, elements, sym) is None


class TestGhostGeometryShape:
    def test_one_plane_yields_one_ghost_copy(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 0, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert len(ghost.elements) == len(elements)

    def test_two_planes_yield_three_ghost_copies(self):
        """2**2 copies minus the real one — the ghosts are everything the
        solver adds that is not in the mesh file."""
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert len(ghost.elements) == 3 * len(elements)

    def test_ghost_nodes_exclude_the_original_block(self):
        """The real mesh is drawn by its own actor; duplicating it in the
        ghost actor would double the drawn surface and darken it."""
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 0, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert len(ghost.nodes) == len(nodes)
        assert not np.allclose(ghost.nodes, nodes)

    def test_ghost_elements_index_into_ghost_nodes(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert ghost.elements.min() >= 0
        assert ghost.elements.max() < len(ghost.nodes)

    def test_ghosts_span_the_reflected_quadrants(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert np.allclose(ghost.nodes.min(axis=0), [-1.0, -1.0, 0.0])

    def test_reflects_about_the_declared_plane_position(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 0, 0], "coordinates": [2.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        # x -> 4 - x, so [0, 1] mirrors to [3, 4]
        assert np.allclose(ghost.nodes[:, 0].min(), 3.0)
        assert np.allclose(ghost.nodes[:, 0].max(), 4.0)


class TestGhostSourceIndex:
    """Every ghost element carries the index of the element it reflects, so
    the viewers can look up its BC colour or its surface scalar."""

    def test_source_index_has_one_entry_per_ghost_element(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert len(ghost.source) == len(ghost.elements)

    def test_source_index_tiles_the_original_elements_per_copy(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        np.testing.assert_array_equal(ghost.source, [0, 1, 0, 1, 0, 1])

    def test_source_index_selects_the_matching_scalar(self):
        """The point of the index: scalars[source] must line up element by
        element with the ghost triangles."""
        nodes, elements = _two_triangles()
        sym = {"planes": [1, 0, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        scalars = np.array([10.0, 20.0])
        np.testing.assert_array_equal(scalars[ghost.source], [10.0, 20.0])


class TestGhostSign:
    """An antisymmetric plane flips the sign of the reflected field; a
    symmetric or reflecting-hard plane does not. Codes 2 and 4 are the two
    NumCalc maps to a negative internal value (NC_3dFunctions.cpp:239)."""

    def _sign(self, planes):
        nodes, elements = _two_triangles()
        sym = {"planes": planes, "coordinates": [0.0, 0.0, 0.0]}
        return ghost_geometry(nodes, elements, sym).sign

    def test_symmetric_plane_keeps_the_sign(self):
        np.testing.assert_array_equal(self._sign([1, 0, 0]), [1.0, 1.0])

    def test_hard_reflecting_plane_keeps_the_sign(self):
        np.testing.assert_array_equal(self._sign([3, 0, 0]), [1.0, 1.0])

    def test_antisymmetric_plane_flips_the_sign(self):
        np.testing.assert_array_equal(self._sign([2, 0, 0]), [-1.0, -1.0])

    def test_soft_reflecting_plane_flips_the_sign(self):
        np.testing.assert_array_equal(self._sign([4, 0, 0]), [-1.0, -1.0])

    def test_sign_multiplies_over_the_planes_each_copy_crossed(self):
        """With one symmetric and one antisymmetric plane the four copies are
        (none), x, y, xy — so only the two that cross y flip."""
        sign = self._sign([1, 2, 0])
        np.testing.assert_array_equal(sign, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

    def test_two_antisymmetric_planes_cancel_on_the_diagonal_copy(self):
        sign = self._sign([2, 2, 0])
        # copies: x (flip), y (flip), xy (flip twice -> back to +1)
        np.testing.assert_array_equal(sign, [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0])

    def test_sign_has_one_entry_per_ghost_element(self):
        nodes, elements = _two_triangles()
        sym = {"planes": [2, 1, 0], "coordinates": [0.0, 0.0, 0.0]}
        ghost = ghost_geometry(nodes, elements, sym)
        assert len(ghost.sign) == len(ghost.elements)


class TestDescribeSymmetry:
    """D8: the rendering is identical for symmetric and reflecting planes, but
    what the ghosts *mean* is not — under a reflecting plane the faded copy is
    a virtual image of a baffle, not part of the object. Label it."""

    def test_returns_none_without_symmetry(self):
        assert describe_symmetry(None) is None

    def test_returns_none_when_every_plane_is_inactive(self):
        assert describe_symmetry({"planes": [0, 0, 0],
                                  "coordinates": [0.0, 0.0, 0.0]}) is None

    def test_names_the_active_axis_and_plane_position(self):
        text = describe_symmetry({"planes": [1, 0, 0],
                                  "coordinates": [0.25, 0.0, 0.0]})
        assert "x = 0.25" in text

    def test_calls_a_symmetric_plane_a_mirrored_half(self):
        text = describe_symmetry({"planes": [1, 0, 0],
                                  "coordinates": [0.0, 0.0, 0.0]})
        assert "mirrored half" in text

    def test_calls_a_reflecting_plane_a_virtual_image(self):
        text = describe_symmetry({"planes": [3, 0, 0],
                                  "coordinates": [0.0, 0.0, 0.0]})
        assert "virtual image" in text

    def test_marks_an_antisymmetric_plane_as_such(self):
        text = describe_symmetry({"planes": [2, 0, 0],
                                  "coordinates": [0.0, 0.0, 0.0]})
        assert "antisymmetric" in text

    def test_reports_the_ghost_count(self):
        text = describe_symmetry({"planes": [1, 1, 0],
                                  "coordinates": [0.0, 0.0, 0.0]})
        assert "3" in text

    def test_describes_every_active_plane(self):
        text = describe_symmetry({"planes": [1, 2, 0],
                                  "coordinates": [0.0, 0.0, 0.0]})
        assert "x = 0" in text and "y = 0" in text
