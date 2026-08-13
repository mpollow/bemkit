"""Loading a mesh file yields a Mesh in metres, whatever the file's units."""

from pathlib import Path

import meshio
import numpy as np
import pytest

from bemkit.geometry import load_mesh
from bemkit.model import Mesh

VERT = np.array([[0., 0, 0], [1000., 0, 0], [0, 1000., 0], [0, 0, 1000.]])
TRI = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])

FIXTURE_MSH = (Path(__file__).resolve().parents[4]
               / "models" / "sphcap" / "mesh" / "coarse.msh")


def test_msh_is_millimetres_and_becomes_metres(tmp_path):
    p = tmp_path / "m.msh"
    meshio.write_points_cells(p, VERT, [("triangle", TRI)], file_format="gmsh")
    mesh = load_mesh(p)
    assert isinstance(mesh, Mesh)
    np.testing.assert_allclose(mesh.vertices.max(), 1.0)


def test_stl_is_already_metres(tmp_path):
    p = tmp_path / "m.stl"
    meshio.write_points_cells(p, VERT / 1000.0, [("triangle", TRI)])
    np.testing.assert_allclose(load_mesh(p).vertices.max(), 1.0)


def test_explicit_scale_overrides_the_extension(tmp_path):
    p = tmp_path / "m.stl"
    meshio.write_points_cells(p, VERT, [("triangle", TRI)])
    np.testing.assert_allclose(load_mesh(p, scale=0.001).vertices.max(), 1.0)


def test_unknown_extension_raises(tmp_path):
    p = tmp_path / "m.wat"
    p.write_text("")
    with pytest.raises(ValueError, match="scale"):
        load_mesh(p)


def test_no_triangle_elements_raises_clearly(tmp_path):
    p = tmp_path / "m.msh"
    # A single line element, no triangles.
    meshio.write_points_cells(
        p, VERT, [("line", np.array([[0, 1]]))], file_format="gmsh")
    with pytest.raises(ValueError, match="triangle"):
        load_mesh(p)


@pytest.mark.skipif(not FIXTURE_MSH.exists(),
                     reason="example fixture mesh not present")
def test_physical_groups_survive_into_mesh_groups():
    """A real gmsh-written .msh carries physical groups across its per-group
    cell blocks; load_mesh must recover them as Mesh.groups so that
    Group("vibrating") selection has something to select."""
    mesh = load_mesh(FIXTURE_MSH)
    assert "vibrating" in mesh.groups
    assert "rigid" in mesh.groups
    vib = mesh.groups["vibrating"]
    assert vib.dtype != bool
    assert len(vib) > 0
    assert vib.max() < mesh.n_elements
    # groups must partition (non-overlapping) and cover every element here
    rig = mesh.groups["rigid"]
    assert len(np.intersect1d(vib, rig)) == 0
    assert len(vib) + len(rig) == mesh.n_elements
