"""bemkit.solvers.fieldpoints: validation + combine (offset bookkeeping)."""

import numpy as np
import pytest

from bemkit.model import Mesh
from bemkit.solvers.fieldpoints import combine, validate

# A unit cube-ish closed tetrahedron centred near the origin, big enough that
# points a few units away are unambiguously outside.
VERT = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.866, 0.0],
    [0.5, 0.289, 0.816],
])
TRI = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [1, 2, 3],
    [0, 2, 3],
])
MESH = Mesh(VERT, TRI)


# -- combine: offsets after filtering ----------------------------------------

def test_combine_with_no_mask_offsets_index_the_concatenation_in_order():
    fields = {"a": np.zeros((2, 3)), "b": np.ones((3, 3))}
    points, offsets = combine(fields)
    assert len(points) == 5
    np.testing.assert_array_equal(offsets["a"], [0, 1])
    np.testing.assert_array_equal(offsets["b"], [2, 3, 4])
    np.testing.assert_array_equal(points[offsets["a"]], fields["a"])
    np.testing.assert_array_equal(points[offsets["b"]], fields["b"])


def test_combine_offsets_stay_correct_when_an_earlier_set_loses_points():
    """The bug most likely to silently mislabel every result: a point from
    set A dropped must not shift where set B's rows land."""
    a = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0]])  # middle point dropped
    b = np.array([[10., 0, 0], [11., 0, 0]])            # both kept
    fields = {"a": a, "b": b}
    # mask over the concatenated [a0, a1, a2, b0, b1] array: drop a1 only.
    mask = np.array([True, False, True, True, True])

    points, offsets = combine(fields, mask)

    assert len(points) == 4
    # 'a' kept its 1st and 3rd original point, now at filtered rows 0, 1
    np.testing.assert_array_equal(offsets["a"], [0, 1])
    np.testing.assert_array_equal(points[offsets["a"]], [a[0], a[2]])
    # 'b' is untouched by a's drop -- must land at rows 2, 3, not 1, 2
    np.testing.assert_array_equal(offsets["b"], [2, 3])
    np.testing.assert_array_equal(points[offsets["b"]], b)


def test_combine_offsets_stay_correct_when_a_later_set_loses_points():
    a = np.array([[0., 0, 0], [1., 0, 0]])
    b = np.array([[10., 0, 0], [11., 0, 0], [12., 0, 0]])  # middle dropped
    fields = {"a": a, "b": b}
    mask = np.array([True, True, True, False, True])

    points, offsets = combine(fields, mask)

    np.testing.assert_array_equal(offsets["a"], [0, 1])
    np.testing.assert_array_equal(points[offsets["a"]], a)
    np.testing.assert_array_equal(offsets["b"], [2, 3])
    np.testing.assert_array_equal(points[offsets["b"]], [b[0], b[2]])


def test_combine_drops_a_set_entirely_when_every_point_of_it_fails():
    a = np.array([[0., 0, 0]])
    b = np.array([[1., 0, 0], [2., 0, 0]])
    mask = np.array([False, True, True])
    points, offsets = combine({"a": a, "b": b}, mask)
    assert len(points) == 2
    assert offsets["a"].tolist() == []
    np.testing.assert_array_equal(offsets["b"], [0, 1])


def test_combine_mask_wrong_length_raises():
    with pytest.raises(ValueError):
        combine({"a": np.zeros((2, 3))}, mask=np.array([True]))


def test_combine_empty_fields_dict():
    points, offsets = combine({})
    assert points.shape == (0, 3)
    assert offsets == {}


# -- validate: geometry --------------------------------------------------

def test_validate_keeps_points_clearly_outside_for_exterior_problem():
    far = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    mask = validate(far, MESH, exterior=True)
    assert mask.all()


def test_validate_drops_points_inside_for_exterior_problem():
    centroid = VERT.mean(axis=0, keepdims=True)
    mask = validate(centroid, MESH, exterior=True)
    assert not mask.any()


def test_validate_interior_problem_flips_the_side():
    centroid = VERT.mean(axis=0, keepdims=True)
    far = np.array([[10.0, 10.0, 10.0]])
    pts = np.vstack([centroid, far])
    mask = validate(pts, MESH, exterior=False)
    assert mask[0] and not mask[1]


def test_validate_drops_points_too_close_to_the_surface():
    # A point that is technically outside but a hair off the surface.
    on_face = VERT[[0, 1, 2]].mean(axis=0)
    normal = np.cross(VERT[1] - VERT[0], VERT[2] - VERT[0])
    normal /= np.linalg.norm(normal)
    just_outside = on_face + normal * 1e-6
    mask = validate(just_outside[None, :], MESH, exterior=True)
    assert not mask.any()
