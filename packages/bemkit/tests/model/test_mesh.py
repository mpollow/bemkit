"""Mesh and Medium: pure data, always metres."""

import numpy as np
import pytest

from bemkit.model import Medium, Mesh

TRI = np.array([[0, 1, 2], [0, 1, 3]])
VERT = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_mesh_holds_arrays_and_groups():
    m = Mesh(VERT, TRI, groups={"vibrating": np.array([0])})
    assert m.n_nodes == 4 and m.n_elements == 2
    np.testing.assert_array_equal(m.groups["vibrating"], [0])


def test_centroids():
    m = Mesh(VERT, TRI)
    np.testing.assert_allclose(m.centroids[0], VERT[[0, 1, 2]].mean(axis=0))


def test_areas():
    m = Mesh(VERT, TRI)
    assert m.areas.shape == (2,)
    assert np.all(m.areas > 0)


def test_rejects_ragged_triangles():
    with pytest.raises(ValueError, match="triangles must be"):
        Mesh(VERT, np.array([[0, 1], [2, 3]]))


def test_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        Mesh(VERT, np.array([[0, 1, 99]]))


def test_unknown_group_is_rejected_at_construction():
    with pytest.raises(ValueError, match="out of range"):
        Mesh(VERT, TRI, groups={"bad": np.array([5])})


def test_medium_defaults_and_impedance():
    med = Medium()
    assert (med.c, med.rho) == (343.0, 1.2)
    assert med.z0 == pytest.approx(343.0 * 1.2)
