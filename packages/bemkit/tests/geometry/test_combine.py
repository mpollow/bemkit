"""Assembling bodies into one Mesh, and holding them apart."""

import numpy as np
import pytest

from bemkit.geometry.combine import (concat_meshes, min_clearance, push_clear,
                                     shells, signed_distance_to_mesh, submesh,
                                     weld)
from bemkit.model import Mesh

from .meshes import sphere, spin, unit_cube, unweld


def _cube(offset=(0.0, 0.0, 0.0), groups=None):
    v, t = unit_cube()
    return Mesh(v + np.asarray(offset, dtype=float), t, groups)


def _sphere(center=(0.0, 0.0, 0.0), radius=1.0, subdiv=3):
    return Mesh(*sphere(center, radius, subdiv))


# --------------------------------------------------------------- concat_meshes

def test_concat_stacks_counts_and_keeps_the_first_body_indexed_from_zero():
    head, glasses = _cube(), _cube(offset=(5, 0, 0))
    out = concat_meshes(head=head, glasses=glasses)

    assert out.n_nodes == head.n_nodes + glasses.n_nodes
    assert out.n_elements == head.n_elements + glasses.n_elements
    #: the whole point of putting the head first — its element numbering is
    #: untouched by whatever is bolted on after it
    np.testing.assert_array_equal(out.triangles[:head.n_elements],
                                  head.triangles)


def test_concat_offsets_the_second_body_onto_its_own_vertices():
    head, glasses = _cube(), _cube(offset=(5, 0, 0))
    out = concat_meshes(head=head, glasses=glasses)

    # Reconstructed geometry, not just indices: an off-by-one in the offset
    # produces a valid Mesh that silently references the wrong body.
    tail = out.triangles[head.n_elements:]
    np.testing.assert_allclose(out.vertices[tail], glasses.vertices[glasses.triangles])


def test_concat_names_each_body_and_namespaces_the_groups_it_brought():
    head = _cube(groups={"pinna": [0, 1]})
    glasses = _cube(offset=(5, 0, 0), groups={"pinna": [2]})
    out = concat_meshes(head=head, glasses=glasses)

    assert set(out.groups) == {"head", "glasses", "head:pinna", "glasses:pinna"}
    np.testing.assert_array_equal(out.groups["head:pinna"], [0, 1])
    #: same group name on both bodies must not collide, and the second body's
    #: indices are rebased onto the combined numbering
    np.testing.assert_array_equal(out.groups["glasses:pinna"],
                                  [head.n_elements + 2])
    np.testing.assert_array_equal(out.groups["glasses"],
                                  np.arange(head.n_elements, out.n_elements))


def test_concat_of_one_body_is_that_body_plus_its_name():
    out = concat_meshes(head=_cube(groups={"pinna": [0]}))
    assert set(out.groups) == {"head", "head:pinna"}


def test_concat_rejects_no_bodies():
    with pytest.raises(ValueError, match="at least one body"):
        concat_meshes()


def test_concat_rejects_a_name_that_would_make_groups_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        concat_meshes(**{"head:left": _cube()})


def test_concat_accepts_names_that_are_not_identifiers():
    out = concat_meshes(**{"earbud left": _cube()})
    assert "earbud left" in out.groups


# ------------------------------------------------------------------------ weld

def test_weld_collapses_per_facet_vertices_without_moving_the_surface():
    v, t = unit_cube()
    loose = Mesh(*unweld(v, t), {"face": [0, 1]})
    assert loose.n_nodes == 3 * len(t)

    fused = weld(loose)
    assert fused.n_nodes == len(v)
    #: element order and groups survive — welding changes connectivity only
    assert fused.n_elements == loose.n_elements
    np.testing.assert_array_equal(fused.groups["face"], [0, 1])
    np.testing.assert_allclose(np.sort(fused.vertices, axis=0),
                               np.sort(v, axis=0))


def test_weld_preserves_element_geometry():
    v, t = unit_cube()
    fused = weld(Mesh(*unweld(v, t)))
    np.testing.assert_allclose(fused.vertices[fused.triangles], v[t])


def test_weld_rejects_a_non_positive_tolerance():
    with pytest.raises(ValueError, match="tol must be positive"):
        weld(_cube(), tol=0.0)


# ---------------------------------------------------------------------- shells

def test_shells_separates_two_disjoint_bodies_largest_first():
    big, small = _sphere(subdiv=2), _sphere(center=(10, 0, 0), subdiv=1)
    out = concat_meshes(big=big, small=small)

    found = shells(out)
    assert len(found) == 2
    assert len(found[0]) == big.n_elements
    np.testing.assert_array_equal(found[0], out.groups["big"])
    np.testing.assert_array_equal(found[1], out.groups["small"])


def test_shells_needs_a_tolerance_for_an_unwelded_mesh():
    """The trap this argument exists for: an STL has no shared corners, so
    every facet reads as its own body until the vertices are welded."""
    v, t = unit_cube()
    loose = Mesh(*unweld(v, t))
    assert len(shells(loose)) == len(t)
    assert len(shells(loose, tol=1e-9)) == 1


def test_shells_of_an_empty_mesh_is_empty():
    assert shells(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))) == []


# --------------------------------------------------------------------- submesh

def test_submesh_extracts_elements_and_drops_unused_vertices():
    cube = _cube()
    out = submesh(cube, [0, 1])

    assert out.n_elements == 2
    assert out.n_nodes == 4                       # the z = 0 face has 4 corners
    np.testing.assert_allclose(out.vertices[out.triangles],
                               cube.vertices[cube.triangles[[0, 1]]])


def test_submesh_restricts_and_reindexes_groups_dropping_the_empty_ones():
    cube = _cube(groups={"kept": [0, 5], "gone": [7]})
    out = submesh(cube, [5, 0])

    assert set(out.groups) == {"kept"}
    #: element 5 became 0 and element 0 became 1, so the group is {0, 1}
    np.testing.assert_array_equal(out.groups["kept"], [0, 1])


def test_submesh_round_trips_a_shell_out_of_a_concatenation():
    head, glasses = _sphere(subdiv=2), _sphere(center=(10, 0, 0), subdiv=1)
    out = concat_meshes(head=head, glasses=glasses)

    back = submesh(out, out.groups["glasses"])
    assert back.n_elements == glasses.n_elements
    np.testing.assert_allclose(back.vertices[back.triangles],
                               glasses.vertices[glasses.triangles])


# ------------------------------------------------------------------- clearance

def test_signed_distance_is_negative_inside_and_positive_outside():
    ball = _sphere(radius=1.0, subdiv=3)
    d = signed_distance_to_mesh([[0, 0, 0], [0, 0, 2.0]], ball)
    assert d[0] < 0
    assert d[1] > 0
    np.testing.assert_allclose(d[1], 1.0, atol=0.02)


def test_min_clearance_measures_the_gap_between_two_bodies():
    a = _sphere(radius=1.0, subdiv=3)
    b = _sphere(center=(3.0, 0, 0), radius=1.0, subdiv=3)
    #: centres 3 apart, radii 1 each -> a 1.0 m gap, up to faceting
    assert min_clearance(a, b) == pytest.approx(1.0, abs=0.02)


def test_min_clearance_uses_faces_not_just_vertices():
    """A coarse body's vertices can all be far from a fine body's vertices
    while a face passes right through the gap. Vertex-to-vertex distance would
    overestimate the clearance — the direction that hides a problem."""
    fine = _sphere(radius=1.0, subdiv=4)
    # Spun off-axis so the two bodies do not present each other a vertex: the
    # nearest feature on the coarse sphere is now the middle of a facet.
    coarse = Mesh(*sphere(center=(2.1, 0, 0), radius=1.0, subdiv=1))
    coarse = Mesh(spin(coarse.vertices, (2.1, 0, 0)), coarse.triangles)

    from scipy.spatial import cKDTree
    vertex_only = cKDTree(coarse.vertices).query(fine.vertices)[0].min()
    assert min_clearance(fine, coarse) < vertex_only - 1e-6


def test_min_clearance_is_symmetric():
    a = _sphere(radius=1.0, subdiv=3)
    b = _sphere(center=(2.5, 0, 0), radius=1.0, subdiv=1)
    assert min_clearance(a, b) == pytest.approx(min_clearance(b, a))


# ------------------------------------------------------------------ push_clear

def test_push_clear_opens_a_gap_between_interpenetrating_bodies():
    head = _sphere(radius=1.0, subdiv=4)
    #: overlapping by 0.1 m, the same shallow interpenetration the glasses have
    device = _sphere(center=(1.9, 0, 0), radius=1.0, subdiv=4)
    #: signed distance, not min_clearance: two surfaces that cross have a true
    #: separation of zero along the intersection curve, but the nearest
    #: *vertex* to that curve is up to half an edge away, so the unsigned
    #: measure never reads zero. Penetration is what we mean here.
    assert signed_distance_to_mesh(device.vertices, head).min() < -0.05

    out = push_clear(device, head, gap=0.05)
    #: The guarantee is at vertices, so the achieved surface clearance falls
    #: short by roughly one facet sagitta, h²/8 — measured 0.0018 here against
    #: a 0.05 gap. That ratio is what makes the vertex guarantee usable in
    #: practice: real device meshes are refined relative to their curvature,
    #: so the sagitta is orders of magnitude below the gap. Refine the fixture
    #: and this margin shrinks; coarsen it and it grows.
    assert min_clearance(head, out) > 0.95 * 0.05


def test_push_clear_keeps_connectivity_and_groups():
    head = _sphere(radius=1.0, subdiv=4)
    device = Mesh(*sphere(center=(1.9, 0, 0), radius=1.0, subdiv=3),
                  {"nozzle": [0, 1, 2]})

    out = push_clear(device, head, gap=0.05)
    np.testing.assert_array_equal(out.triangles, device.triangles)
    np.testing.assert_array_equal(out.groups["nozzle"], [0, 1, 2])
    assert out.n_nodes == device.n_nodes


def test_push_clear_leaves_a_body_that_already_clears_alone():
    head = _sphere(radius=1.0, subdiv=3)
    device = _sphere(center=(5.0, 0, 0), radius=1.0, subdiv=2)
    out = push_clear(device, head, gap=0.05)
    np.testing.assert_array_equal(out.vertices, device.vertices)


def test_push_clear_distortion_is_bounded_by_the_penetration_plus_gap():
    """The displacement has to be reportable and sane — a runaway morph would
    silently turn the device into something else."""
    head = _sphere(radius=1.0, subdiv=4)
    device = _sphere(center=(1.9, 0, 0), radius=1.0, subdiv=3)

    out = push_clear(device, head, gap=0.05)
    moved = np.linalg.norm(out.vertices - device.vertices, axis=1).max()
    assert moved < 0.1 + 0.05 + 0.02       # penetration + gap + faceting


def test_push_clear_rejects_a_non_positive_gap():
    head, device = _sphere(subdiv=2), _sphere(center=(1.9, 0, 0), subdiv=2)
    with pytest.raises(ValueError, match="gap must be positive"):
        push_clear(device, head, gap=0.0)


def test_push_clear_reports_what_it_achieved_when_it_cannot_converge():
    head = _sphere(radius=1.0, subdiv=3)
    device = _sphere(center=(1.9, 0, 0), radius=1.0, subdiv=2)
    with pytest.raises(RuntimeError, match="best clearance"):
        push_clear(device, head, gap=0.05, max_iter=1, smoothing=0)
