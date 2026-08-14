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


# -- ray degeneracy on axis-aligned geometry ---------------------------------

def _axis_aligned_box(half=1.0):
    """A box whose faces, edges and vertices all lie on the coordinate axes —
    the shape that used to defeat the inside test."""
    s = [-half, half]
    v = np.array([[x, y, z] for x in s for y in s for z in s], dtype=float)
    t = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
                  [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
                  [1, 3, 7], [1, 7, 5], [0, 4, 6], [0, 6, 2]])
    return v, t


def test_inside_test_survives_rays_aimed_down_the_axes_of_an_aligned_box():
    """Regression: the ray directions were +x/+y/+z, so for axis-aligned
    geometry every ray from an on-axis point struck vertices and edges exactly.
    Each such hit is counted twice or not at all, flipping the parity — and all
    three votes failed together, so the majority could not recover. The centre
    of a box then read as *outside*, and a field point there would have been
    silently kept for an exterior problem.
    """
    from bemkit.solvers.fieldpoints import points_inside_mesh

    v, t = _axis_aligned_box()
    probes = np.array([[0.0, 0.0, 0.0],      # dead centre, on all three axes
                       [0.5, 0.0, 0.0],      # on the x axis
                       [0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.5],
                       [0.9, 0.9, 0.9]])     # inside, near a corner
    assert points_inside_mesh(probes, v, t).all()


def test_points_outside_an_aligned_box_still_read_as_outside():
    from bemkit.solvers.fieldpoints import points_inside_mesh

    v, t = _axis_aligned_box()
    probes = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0],
                       [1.1, 1.1, 1.1], [-3.0, 0.0, 0.0]])
    assert not points_inside_mesh(probes, v, t).any()


def test_validate_drops_an_on_axis_point_inside_an_aligned_box():
    """The consequence in the layer that matters: models/sphcap places field
    points with on_axis('z', ...), straight down an axis of a mesh whose poles
    sit on that same axis."""
    v, t = _axis_aligned_box()
    points = np.array([[0.0, 0.0, 0.0],     # inside — must be dropped
                       [0.0, 0.0, 5.0]])    # outside on the same axis — kept
    np.testing.assert_array_equal(
        validate(points, Mesh(v, t), exterior=True), [False, True])


# -- validation memo ---------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_validation_cache():
    """Every test in this module starts with an empty memo."""
    from bemkit.solvers.fieldpoints import clear_validation_cache
    clear_validation_cache()
    yield
    clear_validation_cache()


def _probes():
    return np.array([[0.5, 0.3, 0.2], [5.0, 5.0, 5.0], [0.4, 0.3, 0.6]])


def test_validate_repeated_call_gives_the_same_answer():
    a = validate(_probes(), MESH)
    b = validate(_probes(), MESH)
    np.testing.assert_array_equal(a, b)


def test_repeat_is_served_from_the_memo_without_recomputing(monkeypatch):
    """The point of the cache. Exporting a dozen bundles that share one mesh
    and one field-point set asks this identical question a dozen times; on
    models/glassesonme_v2 that was 331 s each, 8645 points against 36262
    triangles."""
    import bemkit.solvers.fieldpoints as fp

    first = validate(_probes(), MESH)

    def explode(*a, **k):
        raise AssertionError("recomputed a cached validation")

    monkeypatch.setattr(fp, "points_inside_mesh", explode)
    monkeypatch.setattr(fp, "min_distance_to_mesh", explode)
    np.testing.assert_array_equal(fp.validate(_probes(), MESH), first)


def test_cache_false_recomputes_and_does_not_populate():
    import bemkit.solvers.fieldpoints as fp

    validate(_probes(), MESH, cache=False)
    assert len(fp._VALIDATE_CACHE) == 0


@pytest.mark.parametrize("change", [
    "points", "vertices", "triangles", "exterior", "min_surface_distance",
])
def test_a_different_input_is_a_different_key(change, monkeypatch):
    """A hit must require bit-identical inputs — otherwise the memo would
    answer for geometry it has never seen, which is the one way a content-keyed
    cache can be wrong."""
    import bemkit.solvers.fieldpoints as fp

    validate(_probes(), MESH)

    pts, mesh, kwargs = _probes(), MESH, {}
    if change == "points":
        pts = pts + 0.001
    elif change == "vertices":
        mesh = Mesh(VERT * 1.0001, TRI)
    elif change == "triangles":
        mesh = Mesh(VERT, TRI[:, ::-1])
    elif change == "exterior":
        kwargs = {"exterior": False}
    else:
        kwargs = {"min_surface_distance": 0.5}

    calls = []
    real = fp.points_inside_mesh
    monkeypatch.setattr(fp, "points_inside_mesh",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    fp.validate(pts, mesh, **kwargs)
    assert calls, f"changing {change} should have forced a recomputation"


def test_the_memo_hands_back_a_copy():
    """A caller that mutates the result must not poison later hits."""
    first = validate(_probes(), MESH)
    first[:] = False
    np.testing.assert_array_equal(validate(_probes(), MESH),
                                  validate(_probes(), MESH, cache=False))


def test_the_memo_is_bounded():
    import bemkit.solvers.fieldpoints as fp

    for i in range(fp._VALIDATE_CACHE_MAX + 5):
        validate(_probes() + i * 0.01, MESH)
    assert len(fp._VALIDATE_CACHE) == fp._VALIDATE_CACHE_MAX


def test_clear_validation_cache_forces_a_recomputation(monkeypatch):
    import bemkit.solvers.fieldpoints as fp

    validate(_probes(), MESH)
    fp.clear_validation_cache()

    calls = []
    real = fp.points_inside_mesh
    monkeypatch.setattr(fp, "points_inside_mesh",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    fp.validate(_probes(), MESH)
    assert calls
