"""Composable per-vertex size fields."""

import numpy as np
import pytest

from bemkit.geometry.metric import (curvature, finest, near_elements,
                                    near_mesh, near_points, sphere_zone,
                                    uniform, wavelength)
from bemkit.model import Mesh

from .meshes import sphere, spin, unit_cube


def _cube():
    return Mesh(*unit_cube())


def _sphere(center=(0.0, 0.0, 0.0), radius=1.0, subdiv=3):
    return Mesh(*sphere(center, radius, subdiv))


# ---------------------------------------------------------------------- finest

def test_finest_takes_the_elementwise_minimum():
    np.testing.assert_array_equal(
        finest([3.0, 1.0, 2.0], [2.0, 5.0, 2.0], [9.0, 9.0, 1.0]),
        [2.0, 1.0, 1.0])


def test_finest_of_one_field_is_that_field():
    np.testing.assert_array_equal(finest([1.0, 2.0]), [1.0, 2.0])


def test_adding_a_term_can_only_refine():
    """The property that makes the terms independent — order-free composition
    and no term needing to know about the others."""
    mesh = _sphere(subdiv=2)
    base = curvature(mesh, 1e-3, 5e-3)
    with_more = finest(base, uniform(mesh, 3e-3))
    assert np.all(with_more <= base)


def test_finest_rejects_fields_of_different_length():
    with pytest.raises(ValueError, match="different meshes"):
        finest(np.ones(4), np.ones(5))


def test_finest_rejects_no_fields():
    with pytest.raises(ValueError, match="at least one"):
        finest()


# --------------------------------------------------------------- simple fields

def test_uniform_is_constant_and_one_per_vertex():
    mesh = _cube()
    s = uniform(mesh, 0.01)
    assert s.shape == (mesh.n_nodes,)
    np.testing.assert_array_equal(s, 0.01)


def test_uniform_rejects_a_non_positive_size():
    with pytest.raises(ValueError, match="h must be positive"):
        uniform(_cube(), 0.0)


def test_wavelength_is_the_band_divided_by_the_element_target():
    mesh = _cube()
    #: 340 / (6 * 16000) = 3.5417 mm — the size 16 kHz actually demands
    np.testing.assert_allclose(
        wavelength(mesh, f_max=16e3, c=340.0), 340.0 / (6 * 16e3))


def test_wavelength_scales_inversely_with_the_element_target():
    mesh = _cube()
    six = wavelength(mesh, 16e3, 340.0, per_wavelength=6)
    three = wavelength(mesh, 16e3, 340.0, per_wavelength=3)
    np.testing.assert_allclose(three, 2.0 * six)


@pytest.mark.parametrize("kwargs", [{"f_max": 0.0}, {"per_wavelength": 0.0}])
def test_wavelength_rejects_non_positive_arguments(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        wavelength(_cube(), **{"f_max": 16e3, **kwargs})


# ------------------------------------------------------------------- curvature

def test_curvature_is_bounded_and_one_per_vertex():
    mesh = _cube()
    s = curvature(mesh, hmin=0.01, hmax=0.1)
    assert s.shape == (mesh.n_nodes,)
    assert np.all(s >= 0.01) and np.all(s <= 0.1)


def test_curvature_of_a_flat_sheet_is_hmax_with_no_divide_by_zero():
    flat = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                np.array([[0, 1, 2], [0, 2, 3]]))
    s = curvature(flat, hmin=0.01, hmax=0.1)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, 0.1)


def test_curvature_refines_a_crease_relative_to_a_flat_face():
    """A box with one face subdivided: the corners are creased, the added
    interior vertices are flat, and the sizes must order accordingly."""
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0.0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                  [0.5, 0.5, 1]])                       # centre of the top face
    t = np.array([[0, 2, 1], [0, 3, 2],
                  [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    s = curvature(Mesh(v, t), hmin=0.01, hmax=0.1)
    assert s[8] > s[0]                       # flat interior coarser than a corner


def test_curvature_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="hmin <= hmax"):
        curvature(_cube(), hmin=0.1, hmax=0.01)


def test_curvature_matches_the_pre_vectorisation_reference():
    """Guards the rewrite from two Python loops to a scatter-add: the metric
    that produced the committed v2 mesh must not change silently."""
    mesh = _sphere(subdiv=2)
    v, t = mesh.vertices, mesh.triangles

    fnr = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    fa = np.linalg.norm(fnr, axis=1) / 2.0
    fn = fnr / (fa[:, None] * 2.0 + 1e-15)
    vn = np.zeros((len(v), 3))
    va = np.zeros(len(v))
    for fi, (a, b, c) in enumerate(t):
        for vi in (a, b, c):
            vn[vi] += fa[fi] * fn[fi]
            va[vi] += fa[fi]
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-15)
    curv = np.zeros(len(v))
    for fi, (a, b, c) in enumerate(t):
        for vi in (a, b, c):
            curv[vi] += fa[fi] * np.arccos(
                np.clip(np.abs(np.dot(fn[fi], vn[vi])), -1.0, 1.0))
    curv /= np.maximum(va, 1e-15)
    lo, hi = curv.min(), curv.max()
    norm = (curv - lo) / (hi - lo) if hi > lo else np.zeros_like(curv)
    expected = np.clip(0.1 - (0.1 - 0.01) * norm, 0.01, 0.1)

    np.testing.assert_allclose(curvature(mesh, 0.01, 0.1), expected, rtol=1e-12)


# ----------------------------------------------------------------- sphere_zone

def test_sphere_zone_clamps_inside_and_leaves_outside_infinite():
    mesh = _cube()
    s = sphere_zone(mesh, center=(0, 0, 0), radius=0.1, h=0.002)
    assert s[0] == 0.002                          # vertex at the origin
    assert np.isinf(s[6])                         # opposite corner


def test_sphere_zone_composes_as_a_pure_refinement():
    mesh = _cube()
    s = finest(uniform(mesh, 0.05),
               sphere_zone(mesh, (0, 0, 0), 0.1, 0.002))
    assert s[0] == 0.002
    assert s[6] == 0.05


def test_sphere_zone_raises_when_it_selects_nothing():
    """The unit trap: metres passed as millimetres puts the anchor a thousand
    times too far away, and a silently-empty zone just means no refinement."""
    with pytest.raises(ValueError, match="no vertices within"):
        sphere_zone(_cube(), center=(500.0, 0, 0), radius=0.1, h=0.002)


# ------------------------------------------------------------ distance ramps

def test_near_points_ramps_from_the_anchor_out_to_h_far():
    mesh = _sphere(radius=1.0, subdiv=3)
    s = near_points(mesh, [[1.0, 0, 0]], h_near=1e-3, h_far=10e-3,
                    d_min=0.0, d_max=0.5)
    #: the anchor sits on a vertex, and the far side of the sphere is 2 m away
    assert s.min() == pytest.approx(1e-3)
    assert s.max() == pytest.approx(10e-3)


def test_near_points_is_flat_when_the_ramp_bounds_are_wide():
    mesh = _sphere(radius=1.0, subdiv=2)
    s = near_points(mesh, [[1.0, 0, 0]], 1e-3, 10e-3, d_min=100.0, d_max=200.0)
    np.testing.assert_allclose(s, 1e-3)


def test_near_elements_measures_from_a_region_of_the_mesh():
    mesh = _sphere(radius=1.0, subdiv=3)
    seed = [0]
    s = near_elements(mesh, seed, 1e-3, 10e-3, 0.0, 1.0)

    on_seed = np.unique(mesh.triangles[seed])
    np.testing.assert_allclose(s[on_seed], 1e-3)
    assert s.max() == pytest.approx(10e-3)


def test_geodesic_and_euclidean_ramps_differ_on_a_folded_surface():
    """The reason ``geodesic`` defaults to True. Two parallel sheets joined at
    one end: the free ends are millimetres apart through space and the full
    length of the fold apart across the surface.
    """
    v = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],       # lower
                  [0.0, 0, 0.01], [1, 0, 0.01],                        # upper
                  [1.0, 1, 0.01], [0, 1, 0.01]])
    t = np.array([[0, 1, 2], [0, 2, 3],          # lower sheet
                  [4, 6, 5], [4, 7, 6],          # upper sheet
                  [1, 5, 6], [1, 6, 2]])         # joined at x = 1
    mesh = Mesh(v, t)
    seed = [0]                                    # on the lower sheet, near x=0

    near = np.unique(mesh.triangles[seed])
    euclid = near_elements(mesh, seed, 1e-3, 10e-3, 0.0, 1.5, geodesic=False)
    geo = near_elements(mesh, seed, 1e-3, 10e-3, 0.0, 1.5, geodesic=True)

    #: vertex 4 sits 10 mm above vertex 0 but is two sheet-lengths away along
    #: the surface, so only the geodesic ramp coarsens it
    assert euclid[4] < geo[4]
    np.testing.assert_allclose(geo[near], 1e-3)


def test_near_elements_rejects_an_empty_region():
    with pytest.raises(ValueError, match="empty element set"):
        near_elements(_cube(), [], 1e-3, 10e-3, 0.0, 1.0)


def test_near_mesh_refines_where_another_body_comes_close():
    head = _sphere(radius=1.0, subdiv=3)
    device = _sphere(center=(2.05, 0, 0), radius=1.0, subdiv=3)

    s = near_mesh(head, device, h_near=1e-3, h_far=10e-3,
                  d_min=0.0, d_max=0.5)

    #: the +x pole faces the device across a 5 cm gap; the -x pole is 2 m away
    facing = np.argmax(head.vertices[:, 0])
    away = np.argmin(head.vertices[:, 0])
    assert s[facing] < s[away]
    assert s[away] == pytest.approx(10e-3)


def test_near_mesh_uses_faces_so_a_coarse_body_cannot_hide():
    """Vertex-to-vertex distance would report the coarse device as further away
    than it is and leave the head under-refined at the junction."""
    head = _sphere(radius=1.0, subdiv=4)
    # Spun so the bodies do not present each other a vertex — otherwise the
    # two measures coincide exactly and the test proves nothing.
    coarse = _sphere(center=(2.05, 0, 0), radius=1.0, subdiv=1)
    coarse = Mesh(spin(coarse.vertices, (2.05, 0, 0)), coarse.triangles)

    from scipy.spatial import cKDTree
    s_faces = near_mesh(head, coarse, 1e-3, 10e-3, 0.0, 0.5)
    d_verts = cKDTree(coarse.vertices).query(head.vertices)[0]
    from bemkit.geometry.geodesic import size_from_distance
    s_verts = size_from_distance(d_verts, 1e-3, 10e-3, 0.0, 0.5)

    assert s_faces.min() < s_verts.min()
