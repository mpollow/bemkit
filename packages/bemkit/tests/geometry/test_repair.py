"""Closing holes so a surface can be a BEM boundary."""

import numpy as np
import pytest

from bemkit.geometry.quality import check_orientation, check_watertight
from bemkit.geometry.repair import boundary_loops, fill_holes
from bemkit.model import Mesh

from .meshes import enclosed_volume, sphere, unit_cube


def _holed_sphere(n_removed=4, subdiv=3):
    """A sphere with a few adjacent faces deleted — one hole."""
    v, t = sphere(subdiv=subdiv)
    # Faces sharing a vertex, so the removal leaves a single loop rather than
    # several scattered ones.
    touching = np.nonzero((t == t[0, 0]).any(axis=1))[0][:n_removed]
    keep = np.setdiff1d(np.arange(len(t)), touching)
    return Mesh(v, t[keep])


def test_a_closed_mesh_has_no_loops():
    assert boundary_loops(Mesh(*sphere(subdiv=2))) == []


def test_filling_a_closed_mesh_changes_nothing():
    mesh = Mesh(*sphere(subdiv=2))
    out = fill_holes(mesh)
    np.testing.assert_array_equal(out.triangles, mesh.triangles)
    np.testing.assert_array_equal(out.vertices, mesh.vertices)


def test_a_hole_is_found_and_described():
    mesh = _holed_sphere()
    loops = boundary_loops(mesh)

    assert len(loops) == 1
    assert len(loops[0]["vertices"]) > 2
    #: a small cap removed from a unit sphere is nearly flat
    assert loops[0]["planarity"] < 0.1


def test_two_holes_are_reported_separately_largest_first():
    v, t = sphere(subdiv=3)
    #: opposite poles, so the two holes cannot merge into one loop
    a = np.nonzero((t == np.argmax(v[:, 0])).any(axis=1))[0][:6]
    b = np.nonzero((t == np.argmin(v[:, 0])).any(axis=1))[0][:3]
    mesh = Mesh(v, t[np.setdiff1d(np.arange(len(t)), np.concatenate([a, b]))])

    loops = boundary_loops(mesh)
    assert len(loops) == 2
    assert len(loops[0]["vertices"]) >= len(loops[1]["vertices"])


def test_filling_closes_the_surface():
    mesh = _holed_sphere()
    assert check_watertight(mesh.vertices, mesh.triangles).level == "error"

    out = fill_holes(mesh)
    assert check_watertight(out.vertices, out.triangles).level == "ok"


def test_the_patch_winds_to_match_the_surface_it_is_sewn_onto():
    """Winding comes from the boundary edge's own direction, so the fill agrees
    with the surface whichever way round that surface is — a fill that guessed
    would leave a closed mesh with inverted faces, which check_watertight
    happily passes."""
    mesh = _holed_sphere()
    out = fill_holes(mesh)
    assert check_orientation(out.vertices, out.triangles).level == "ok"


def test_an_inward_wound_surface_is_filled_inward_too():
    v, t = sphere(subdiv=3)
    flipped = Mesh(v, t[:, ::-1])                     # every normal inverted
    holed = Mesh(v, flipped.triangles[4:])

    out = fill_holes(holed)
    assert check_watertight(out.vertices, out.triangles).level == "ok"
    #: still consistently wound — just consistently the wrong way, which is a
    #: separate problem the orientation check names
    assert enclosed_volume(out.vertices, out.triangles) < 0


def test_filling_preserves_existing_indices_and_groups():
    """The fill is appended, so anything already frozen or grouped stays valid."""
    mesh = _holed_sphere()
    mesh = Mesh(mesh.vertices, mesh.triangles, {"keep": [0, 1, 2]})

    out = fill_holes(mesh)
    np.testing.assert_array_equal(out.triangles[:mesh.n_elements],
                                  mesh.triangles)
    np.testing.assert_array_equal(out.vertices[:mesh.n_nodes], mesh.vertices)
    np.testing.assert_array_equal(out.groups["keep"], [0, 1, 2])


def test_filled_volume_is_close_to_the_original():
    v, t = sphere(subdiv=4)
    full = enclosed_volume(v, t)
    holed = Mesh(v, t[8:])
    out = fill_holes(holed)
    assert enclosed_volume(out.vertices, out.triangles) == pytest.approx(
        full, rel=0.01)


def test_a_strongly_non_planar_loop_is_refused():
    """A fan from one centroid would invent a cone. Better to say so than to
    produce a closed surface that is confidently the wrong shape."""
    #: an open cylinder — its two rims are planar, but the whole side boundary
    #: of a half-cylinder is not
    ang = np.linspace(0, 1.6 * np.pi, 16)
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    top = ring + [0, 0, 2.0]
    v = np.vstack([ring, top])
    n = len(ring)
    t = []
    for i in range(n - 1):
        t += [[i, i + 1, n + i], [i + 1, n + i + 1, n + i]]
    mesh = Mesh(v, np.array(t))

    with pytest.raises(ValueError, match="not planar enough"):
        fill_holes(mesh)


def test_a_non_planar_loop_can_be_filled_when_explicitly_allowed():
    ang = np.linspace(0, 1.6 * np.pi, 16)
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])
    v = np.vstack([ring, ring + [0, 0, 2.0]])
    n = len(ring)
    t = [tri for i in range(n - 1)
         for tri in ([i, i + 1, n + i], [i + 1, n + i + 1, n + i])]
    mesh = Mesh(v, np.array(t))

    out = fill_holes(mesh, max_planarity=None)
    assert out.n_elements > mesh.n_elements


def test_an_oversized_boundary_is_refused_as_a_missing_region():
    v, t = sphere(subdiv=3)
    half = Mesh(v, t[t.mean(axis=1) >= 0][: len(t) // 2])
    with pytest.raises(ValueError, match="max_vertices"):
        fill_holes(half, max_planarity=None, max_vertices=4)


def test_a_cube_missing_a_face_is_closed_back_to_its_volume():
    v, t = unit_cube()
    mesh = Mesh(v, t[2:])                              # drop the z = 1 face
    out = fill_holes(mesh)

    assert check_watertight(out.vertices, out.triangles).level == "ok"
    assert check_orientation(out.vertices, out.triangles).level == "ok"
    assert enclosed_volume(out.vertices, out.triangles) == pytest.approx(1.0)
