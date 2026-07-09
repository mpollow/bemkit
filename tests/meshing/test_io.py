"""Tests for pynumcalc.meshing.io — get_mesh_scale, find_mesh_file, MeshInfo."""

import numpy as np
import pytest
from pathlib import Path

from pynumcalc.meshing.io import get_mesh_scale, find_mesh_file, MeshInfo, MESH_EXTENSIONS


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
