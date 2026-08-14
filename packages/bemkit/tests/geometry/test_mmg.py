"""Medit I/O for mmgs, and driving the remesher."""

import shutil

import numpy as np
import pytest

from bemkit.geometry.mmg import (DEFAULT_BINARY, curvature_sizes, read_mesh,
                                 remesh, write_mesh, write_sol)
from bemkit.model import Mesh

from .meshes import enclosed_volume, sphere, unit_cube

needs_mmgs = pytest.mark.skipif(
    shutil.which(DEFAULT_BINARY) is None,
    reason=f"{DEFAULT_BINARY} not on PATH")


def _cube():
    """Flat faces, sharp edges — curvature varies a lot."""
    return Mesh(*unit_cube())


def _sphere(radius=1.0, subdiv=3):
    return Mesh(*sphere(radius=radius, subdiv=subdiv))


# ------------------------------------------------------------- curvature sizes

def test_sizes_are_one_per_vertex_and_within_bounds():
    mesh = _cube()
    s = curvature_sizes(mesh, hmin=0.01, hmax=0.1)
    assert s.shape == (mesh.n_nodes,)
    assert np.all(s >= 0.01) and np.all(s <= 0.1)


def test_flat_mesh_gets_hmax_everywhere_no_divide_by_zero():
    flat = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
                np.array([[0, 1, 2], [0, 2, 3]]))
    s = curvature_sizes(flat, hmin=0.01, hmax=0.1)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s, 0.1)


def test_write_sol_round_trips_the_count(tmp_path):
    mesh = _cube()
    s = curvature_sizes(mesh, hmin=0.01, hmax=0.1)
    p = tmp_path / "m.sol"
    write_sol(p, s)
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert str(mesh.n_nodes) in lines


# ------------------------------------------------------------------- Medit I/O

def test_write_mesh_round_trips_geometry_through_read_mesh(tmp_path):
    mesh = _sphere(subdiv=2)
    p = tmp_path / "m.mesh"
    write_mesh(p, mesh)

    back = read_mesh(p)
    assert back.n_nodes == mesh.n_nodes
    assert back.n_elements == mesh.n_elements
    np.testing.assert_allclose(back.vertices, mesh.vertices)
    np.testing.assert_array_equal(back.triangles, mesh.triangles)


def test_write_mesh_emits_one_based_indices(tmp_path):
    """Medit counts from 1. Off by one here silently rotates every triangle
    onto its neighbour's corners, which still reads back as a valid mesh."""
    mesh = _cube()
    p = tmp_path / "m.mesh"
    write_mesh(p, mesh)

    lines = p.read_text().splitlines()
    start = lines.index("Triangles") + 2
    first = [int(x) for x in lines[start].split()[:3]]
    np.testing.assert_array_equal(np.array(first) - 1, mesh.triangles[0])


def test_refs_survive_the_round_trip_as_groups(tmp_path):
    """References are the only label that survives mmgs, which renumbers both
    vertices and triangles — so a region has to travel as a ref, not an index."""
    mesh = _cube()
    refs = np.ones(mesh.n_elements, dtype=int)
    refs[[0, 1]] = 7
    p = tmp_path / "m.mesh"
    write_mesh(p, mesh, refs=refs)

    back = read_mesh(p)
    assert set(back.groups) == {"ref1", "ref7"}
    np.testing.assert_array_equal(back.groups["ref7"], [0, 1])


def test_write_mesh_rejects_a_ref_array_of_the_wrong_length(tmp_path):
    mesh = _cube()
    with pytest.raises(ValueError, match="refs has"):
        write_mesh(tmp_path / "m.mesh", mesh, refs=np.ones(3, dtype=int))


def test_required_sections_are_written_when_asked(tmp_path):
    mesh = _cube()
    p = tmp_path / "m.mesh"
    write_mesh(p, mesh, required_triangles=[0, 1], required_vertices=[2])

    text = p.read_text()
    assert "RequiredTriangles" in text and "RequiredVertices" in text
    lines = text.splitlines()
    at = lines.index("RequiredTriangles")
    assert lines[at + 1] == "2"
    assert [lines[at + 2], lines[at + 3]] == ["1", "2"]      # 1-based


def test_no_required_sections_when_none_are_requested(tmp_path):
    p = tmp_path / "m.mesh"
    write_mesh(p, _cube())
    assert "Required" not in p.read_text()


def test_an_empty_required_list_writes_no_section(tmp_path):
    """mmg rejects a Required block with a count of zero."""
    p = tmp_path / "m.mesh"
    write_mesh(p, _cube(), required_triangles=[])
    assert "RequiredTriangles" not in p.read_text()


def test_read_mesh_copes_with_required_sections_in_the_file(tmp_path):
    """mmgs echoes Required* blocks into its output, and meshio's Medit reader
    does not know the keyword — if that trips it, every frozen remesh fails on
    the way back in."""
    p = tmp_path / "m.mesh"
    mesh = _cube()
    write_mesh(p, mesh, required_triangles=[0, 1], required_vertices=[2])

    back = read_mesh(p)
    assert back.n_elements == mesh.n_elements
    np.testing.assert_allclose(back.vertices, mesh.vertices)


# -------------------------------------------------------------------- remeshing

def test_remesh_reports_a_missing_binary_rather_than_a_shell_error():
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        remesh(_cube(), binary="mmgs_does_not_exist")


def test_remesh_rejects_a_size_field_of_the_wrong_length():
    mesh = _cube()
    with pytest.raises(ValueError, match="sizes has"):
        remesh(mesh, np.ones(mesh.n_nodes + 1))


@needs_mmgs
def test_remesh_refines_a_sphere_towards_the_requested_size():
    mesh = _sphere(subdiv=2)
    out = remesh(mesh, np.full(mesh.n_nodes, 0.15), hausd=0.01)

    assert out.n_elements > mesh.n_elements
    #: still a sphere: the enclosed volume must not have moved much
    assert enclosed_volume(out.vertices, out.triangles) == pytest.approx(
        4.0 / 3.0 * np.pi, rel=0.05)


@needs_mmgs
def test_remesh_honours_a_graded_size_field():
    """The whole point of a per-vertex metric: one half of the surface comes
    back finer than the other."""
    mesh = _sphere(subdiv=3)
    sizes = np.where(mesh.vertices[:, 0] > 0.0, 0.08, 0.4)
    out = remesh(mesh, sizes, hausd=0.02)

    edge = np.sqrt(4 * out.areas / np.sqrt(3))
    fine = edge[out.centroids[:, 0] > 0.5]
    coarse = edge[out.centroids[:, 0] < -0.5]
    assert fine.mean() < 0.5 * coarse.mean()


@needs_mmgs
def test_remesh_carries_refs_through_to_groups():
    mesh = _sphere(subdiv=3)
    refs = np.where(mesh.centroids[:, 0] > 0.0, 2, 1)
    out = remesh(mesh, np.full(mesh.n_nodes, 0.2), refs=refs, hausd=0.02)

    assert set(out.groups) == {"ref1", "ref2"}
    assert len(out.groups["ref2"]) > 0


def _canonical_triangles(mesh, elements=None):
    """Each triangle as an order-independent tuple of rounded coordinates.

    mmgs renumbers everything, so identity can only be asserted geometrically.
    """
    t = mesh.triangles if elements is None else mesh.triangles[elements]
    return {tuple(sorted(map(tuple, tri)))
            for tri in np.round(mesh.vertices[t], 10)}


@needs_mmgs
def test_required_triangles_come_back_verbatim_while_the_rest_is_refined():
    """The guarantee the whole head-plus-device pipeline rests on.

    One head mesh has to be shared by every device configuration, so remeshing
    the junction must not disturb the ear. mmgs is the only lever for that:
    mark the region required and it is passed through untouched. Verified here
    at the ratio the real case uses — around 85% of the surface frozen — and
    geometrically, since mmgs renumbers vertices and triangles freely.
    """
    mesh = _sphere(subdiv=4)
    free = np.nonzero(mesh.centroids[:, 0] > 0.7)[0]
    frozen = np.setdiff1d(np.arange(mesh.n_elements), free)
    #: pin the corners too — a required *triangle* does not pin its vertices,
    #: and a moved corner silently reshapes a triangle that was meant to be
    #: frozen. Only the interior ones: a vertex on the boundary of the free
    #: zone has to be free for the refinement to attach to anything.
    frozen_verts = np.setdiff1d(np.unique(mesh.triangles[frozen]),
                                np.unique(mesh.triangles[free]))
    assert len(frozen) / mesh.n_elements > 0.8

    out = remesh(mesh, np.full(mesh.n_nodes, 0.03), hausd=0.01,
                 required_triangles=frozen, required_vertices=frozen_verts)

    want, got = _canonical_triangles(mesh, frozen), _canonical_triangles(out)
    assert want <= got, f"{len(want - got)} frozen triangles were modified"

    #: and the free zone really was refined, rather than everything frozen
    assert len(got - want) > 5 * len(free)

    #: every new triangle lies inside the free region — the refinement did not
    #: bleed across the boundary
    new_centroids = np.array([np.mean(t, axis=0) for t in got - want])
    assert new_centroids[:, 0].min() > 0.6


@needs_mmgs
def test_required_vertices_are_preserved_to_full_precision():
    mesh = _sphere(subdiv=3)
    free = np.nonzero(mesh.centroids[:, 0] > 0.7)[0]
    frozen = np.setdiff1d(np.arange(mesh.n_elements), free)
    frozen_verts = np.setdiff1d(np.unique(mesh.triangles[frozen]),
                                np.unique(mesh.triangles[free]))

    out = remesh(mesh, np.full(mesh.n_nodes, 0.05), hausd=0.01,
                 required_triangles=frozen, required_vertices=frozen_verts)

    kept = {tuple(v) for v in np.round(out.vertices, 12)}
    for v in np.round(mesh.vertices[frozen_verts], 12):
        assert tuple(v) in kept


@needs_mmgs
def test_freezing_most_of_the_surface_bounds_how_much_mmgs_adds():
    """Sanity bound on the mechanism: with 85% required, the result must be far
    smaller than remeshing the same size field unconstrained."""
    mesh = _sphere(subdiv=4)
    sizes = np.full(mesh.n_nodes, 0.03)
    free = np.nonzero(mesh.centroids[:, 0] > 0.7)[0]
    frozen = np.setdiff1d(np.arange(mesh.n_elements), free)
    frozen_verts = np.setdiff1d(np.unique(mesh.triangles[frozen]),
                                np.unique(mesh.triangles[free]))

    constrained = remesh(mesh, sizes, hausd=0.01,
                         required_triangles=frozen,
                         required_vertices=frozen_verts)
    unconstrained = remesh(mesh, sizes, hausd=0.01)

    assert constrained.n_elements < 0.3 * unconstrained.n_elements


@needs_mmgs
def test_remesh_reports_mmgs_output_when_it_fails():
    """A negative gradation is rejected by mmgs, not by us — the point is that
    its complaint reaches the caller instead of a bare non-zero exit."""
    mesh = _sphere(subdiv=2)
    with pytest.raises(RuntimeError, match="failed with code"):
        remesh(mesh, np.full(mesh.n_nodes, 0.2), hmin=-1.0, hmax=-2.0)
