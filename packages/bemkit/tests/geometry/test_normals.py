"""Face and vertex normals — the shared area-weighting convention."""

import numpy as np

from bemkit.geometry.normals import face_normals, vertex_areas, vertex_normals

from .meshes import sphere, unit_cube


def test_face_normals_are_unit_and_areas_sum_to_the_surface():
    v, t = unit_cube()
    n, a = face_normals(v, t)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0)
    #: 6 faces of a unit cube, two triangles each
    assert a.sum() == 6.0


def test_face_normals_point_outward_for_ccw_winding():
    v, t = unit_cube()
    n, _ = face_normals(v, t)
    centroids = v[t].mean(axis=1)
    outward = centroids - np.array([0.5, 0.5, 0.5])
    assert np.all(np.einsum("ij,ij->i", n, outward) > 0)


def test_unnormalised_normals_have_twice_the_area_magnitude():
    v, t = unit_cube()
    raw, a = face_normals(v, t, normalise=False)
    np.testing.assert_allclose(np.linalg.norm(raw, axis=1), 2.0 * a)


def test_vertex_normals_on_a_sphere_are_radial():
    """Exact only in the limit — a faceted sphere's vertex normal is the
    area-weighted mean of its incident facets, which lags the true radial by
    O(h²). Tightening this tolerance means refining the sphere, not fixing
    the code."""
    v, t = sphere(radius=2.0, subdiv=4)
    n = vertex_normals(v, t)
    np.testing.assert_allclose(n, v / 2.0, atol=1e-2)


def test_degenerate_face_yields_zero_not_nan():
    """A collapsed triangle has no direction. It must not poison its corners'
    normals with NaN — a single bad facet in a scan would otherwise take out
    the whole curvature metric."""
    v = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    t = np.array([[0, 1, 2]])
    n, a = face_normals(v, t)
    assert a[0] == 0.0
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(vertex_normals(v, t)))


def test_vertex_areas_triple_count_the_surface():
    """Every face contributes its area to each of three corners."""
    v, t = unit_cube()
    _, a = face_normals(v, t)
    np.testing.assert_allclose(vertex_areas(v, t).sum(), 3.0 * a.sum())


def test_vertex_normal_weighting_follows_area_not_face_count():
    """Two coplanar slivers must not outvote one large perpendicular face.

    An unweighted mean over the three incident faces would tilt the normal a
    third of the way toward the slivers' direction; area weighting keeps it
    essentially on the big face.
    """
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0],       # big face, z+
                  [0.0, 0, -1e-3], [1, 0, -1e-3]])          # two slivers, y-
    t = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 4]])
    n = vertex_normals(v, t)[0]
    assert abs(n[2] - 1.0) < 1e-2
