"""Tests for bemkit.meshing.io — get_mesh_scale, find_mesh_file, MeshInfo."""

import numpy as np
import pytest
from pathlib import Path

from bemkit.meshing.io import (
    get_mesh_scale, find_mesh_file, MeshInfo, MESH_EXTENSIONS, mirror_mesh,
)


class TestGetMeshScale:
    def test_msh_is_mm(self):
        assert get_mesh_scale("model.msh") == 0.001

    def test_stl_is_meters(self):
        assert get_mesh_scale("model.stl") == 1.0

    def test_ply_is_meters(self):
        assert get_mesh_scale("model.ply") == 1.0

    def test_mesh_is_meters(self):
        assert get_mesh_scale("model.mesh") == 1.0

    def test_path_object(self):
        assert get_mesh_scale(Path("/some/dir/foo.msh")) == 0.001


class TestFindMeshFile:
    def test_finds_msh(self, tmp_path):
        (tmp_path / "model.msh").touch()
        assert find_mesh_file(tmp_path, "model") == tmp_path / "model.msh"

    def test_finds_stl(self, tmp_path):
        (tmp_path / "model.stl").touch()
        assert find_mesh_file(tmp_path, "model") == tmp_path / "model.stl"

    def test_priority_order(self, tmp_path):
        """msh takes priority over stl."""
        (tmp_path / "model.msh").touch()
        (tmp_path / "model.stl").touch()
        assert find_mesh_file(tmp_path, "model").suffix == ".msh"

    def test_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No mesh file found"):
            find_mesh_file(tmp_path, "nonexistent")

    def test_full_name_direct_lookup(self, tmp_path):
        """Full filename bypasses priority search."""
        (tmp_path / "model.mesh").touch()
        assert find_mesh_file(tmp_path, "model.mesh") == tmp_path / "model.mesh"

    def test_full_name_overrides_priority(self, tmp_path):
        """Full filename returns specified file even if higher-priority extension exists."""
        (tmp_path / "model.msh").touch()
        (tmp_path / "model.stl").touch()
        (tmp_path / "model.mesh").touch()
        assert find_mesh_file(tmp_path, "model.mesh") == tmp_path / "model.mesh"

    def test_full_name_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Mesh file not found"):
            find_mesh_file(tmp_path, "missing.stl")


class TestMeshInfo:
    def test_from_file_msh(self, tmp_path):
        f = tmp_path / "test.msh"
        f.touch()
        info = MeshInfo.from_file(f)
        assert info.mesh_format == ".msh"
        assert info.scale == 0.001

    def test_from_file_stl(self, tmp_path):
        f = tmp_path / "test.stl"
        f.touch()
        info = MeshInfo.from_file(f)
        assert info.mesh_format == ".stl"
        assert info.scale == 1.0

    def test_frozen(self, tmp_path):
        f = tmp_path / "test.msh"
        f.touch()
        info = MeshInfo.from_file(f)
        with pytest.raises(AttributeError):
            info.scale = 2.0


class TestMirrorMesh:
    """mirror_mesh expands a symmetry-reduced mesh into the full geometry."""

    @staticmethod
    def _quarter_box():
        """Quarter of the box [-1,1] x [-1,1] x [0,1], kept in x,y >= 0.

        Open at the two cut planes (no x=0 or y=0 faces), like a real
        quarter-symmetric mesh, so mirroring produces no duplicate elements.
        """
        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], float)
        t = np.array([
            [0, 2, 1], [0, 3, 2],        # z = 0
            [4, 5, 6], [4, 6, 7],        # z = 1
            [1, 2, 6], [1, 6, 5],        # x = 1
            [3, 7, 6], [3, 6, 2],        # y = 1
        ])
        return v, t

    def test_no_active_planes_returns_mesh_unchanged(self):
        v, t = self._quarter_box()
        vm, tm = mirror_mesh(v, t, [0, 0, 0], [0.0, 0.0, 0.0])
        assert np.array_equal(vm, v)
        assert np.array_equal(tm, t)

    def test_one_plane_doubles_the_element_count(self):
        v, t = self._quarter_box()
        vm, tm = mirror_mesh(v, t, [1, 0, 0], [0.0, 0.0, 0.0])
        assert len(tm) == 2 * len(t)

    def test_two_planes_quadruple_the_element_count(self):
        v, t = self._quarter_box()
        vm, tm = mirror_mesh(v, t, [1, 1, 0], [0.0, 0.0, 0.0])
        assert len(tm) == 4 * len(t)

    def test_mirrored_copy_spans_the_reflected_coordinates(self):
        v, t = self._quarter_box()
        vm, _ = mirror_mesh(v, t, [1, 1, 0], [0.0, 0.0, 0.0])
        assert np.allclose(vm.min(axis=0), [-1, -1, 0])
        assert np.allclose(vm.max(axis=0), [1, 1, 1])

    def test_reflects_about_an_offset_plane_not_the_origin(self):
        v, t = self._quarter_box()
        vm, _ = mirror_mesh(v, t, [1, 0, 0], [2.0, 0.0, 0.0])
        # x -> 2*2 - x, so [0, 1] mirrors to [3, 4]
        assert np.allclose(vm.min(axis=0)[0], 0.0)
        assert np.allclose(vm.max(axis=0)[0], 4.0)

    def test_mirrored_copy_has_reversed_winding(self):
        """Reflection flips orientation; winding must flip back so that
        normals still point out of the mirrored surface."""
        v, t = self._quarter_box()
        vm, tm = mirror_mesh(v, t, [1, 0, 0], [0.0, 0.0, 0.0])
        orig, mirrored = tm[:len(t)], tm[len(t):]

        def normals(tris):
            a, b, c = vm[tris[:, 0]], vm[tris[:, 1]], vm[tris[:, 2]]
            n = np.cross(b - a, c - a)
            return n / np.linalg.norm(n, axis=1, keepdims=True)

        # the z=0 face (first two triangles) faces -z on both copies
        assert np.allclose(normals(orig)[:2][:, 2], -1.0)
        assert np.allclose(normals(mirrored)[:2][:, 2], -1.0)

    def test_rejects_a_planes_list_that_is_not_three_long(self):
        v, t = self._quarter_box()
        with pytest.raises(ValueError, match="three"):
            mirror_mesh(v, t, [1, 1], [0.0, 0.0, 0.0])
