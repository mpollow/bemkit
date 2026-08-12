"""Selectors resolve a Mesh to element indices."""

import numpy as np
import pytest

from bemkit.model import Elements, Group, Mesh, Near, Where

VERT = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
TRI = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
MESH = Mesh(VERT, TRI, groups={"vibrating": np.array([0, 1])})


def test_group_resolves_by_name():
    np.testing.assert_array_equal(Group("vibrating").resolve(MESH), [0, 1])


def test_group_is_case_insensitive():
    """Foreign meshers name groups their own way (Akabak writes SourceDisc)."""
    np.testing.assert_array_equal(Group("VIBRATING").resolve(MESH), [0, 1])


def test_missing_group_names_what_is_available():
    with pytest.raises(KeyError, match="vibrating"):
        Group("nope").resolve(MESH)


def test_elements_passes_indices_through():
    np.testing.assert_array_equal(Elements([2, 0]).resolve(MESH), [0, 2])


def test_elements_rejects_a_boolean_mask():
    """[True, False, True, False] must not silently become [1, 0, 1, 0]."""
    with pytest.raises(ValueError, match="boolean"):
        Elements([True, False, True, False])


def test_near_picks_the_nearest_element():
    target = MESH.centroids[2]
    np.testing.assert_array_equal(Near(target).resolve(MESH), [2])


def test_near_with_radius_selects_all_within():
    got = Near(MESH.centroids[0], radius=10.0).resolve(MESH)
    np.testing.assert_array_equal(got, [0, 1, 2, 3])


def test_near_rejects_an_anchor_far_from_the_mesh():
    """A wrong unit or frame lands the anchor nowhere near the surface."""
    with pytest.raises(ValueError, match="mesh extent"):
        Near((500.0, 0, 0)).resolve(MESH)


def test_near_rejects_a_radius_that_selects_nothing():
    # A radius is given, so the far-mesh-extent check is bypassed entirely;
    # the point just needs to be farther than the radius from every centroid
    # (MESH.centroids[0] itself is 0 m from element 0 and would wrongly pass).
    with pytest.raises(ValueError, match="no element"):
        Near(MESH.centroids[0] + 1.0, radius=1e-9).resolve(MESH)


def test_where_takes_a_predicate_on_centroids():
    got = Where(lambda c: c[:, 2] > 0.2).resolve(MESH)
    assert set(got) == {1, 2, 3}


def test_where_rejecting_everything_raises():
    with pytest.raises(ValueError, match="no element"):
        Where(lambda c: c[:, 0] > 1e6).resolve(MESH)


def test_where_rejects_a_non_boolean_return():
    """An index array must not be silently reinterpreted as a boolean mask."""
    with pytest.raises(TypeError, match="boolean"):
        Where(lambda c: np.array([1, 2])).resolve(MESH)


def test_where_rejects_a_mask_of_the_wrong_length():
    with pytest.raises(ValueError, match="shape"):
        Where(lambda c: np.array([True, False])).resolve(MESH)
