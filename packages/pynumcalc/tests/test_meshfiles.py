"""Tests for pynumcalc.meshfiles — write→read verification.

These writers take arrays, never mesh-file paths: that is what keeps
pynumcalc free of meshio and of bemkit's mm/m convention.
"""

import numpy as np
import pytest

from pynumcalc.meshfiles import (
    write_eval_grid_elements,
    write_eval_grid_nodes,
    write_object_mesh,
)
from pynumcalc.readers import read_eval_grid_nodes


def _count(path):
    """First line of a NumCalc Nodes.txt/Elements.txt is its count."""
    return int(path.read_text().splitlines()[0])


class TestWriteEvalGridNodes:
    def test_roundtrip(self, tmp_path):
        pts = np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        write_eval_grid_nodes(pts, tmp_path)
        np.testing.assert_allclose(read_eval_grid_nodes(tmp_path), pts)

    def test_creates_directory(self, tmp_path):
        write_eval_grid_nodes(np.array([[0.0, 0.0, 0.0]]), tmp_path / "sub")
        assert (tmp_path / "sub" / "EvalGrid" / "Nodes.txt").exists()

    def test_node_count_line(self, tmp_path):
        write_eval_grid_nodes(np.zeros((5, 3)), tmp_path)
        assert _count(tmp_path / "EvalGrid" / "Nodes.txt") == 5

    def test_custom_start_id(self, tmp_path):
        pts = np.array([[1.0, 2.0, 3.0]])
        write_eval_grid_nodes(pts, tmp_path, node_id_start=100)
        # Should still roundtrip correctly (IDs are metadata)
        np.testing.assert_allclose(read_eval_grid_nodes(tmp_path), pts)


class TestWriteEvalGridElements:
    def test_creates_file(self, tmp_path):
        write_eval_grid_elements(5, tmp_path)
        assert (tmp_path / "EvalGrid" / "Elements.txt").exists()

    def test_element_count(self, tmp_path):
        write_eval_grid_elements(10, tmp_path)
        assert _count(tmp_path / "EvalGrid" / "Elements.txt") == 9  # n-1 segments

    def test_single_point(self, tmp_path):
        write_eval_grid_elements(1, tmp_path)
        assert _count(tmp_path / "EvalGrid" / "Elements.txt") == 1  # max(1, 0)

    def test_zero_points_raises(self, tmp_path):
        # A zero-point EvalGrid would write an element referencing an
        # undeclared node — reject instead of producing a corrupt bundle.
        with pytest.raises(ValueError):
            write_eval_grid_elements(0, tmp_path)


class TestWriteObjectMesh:
    def test_counts_and_roundtrip(self, tmp_path):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tris = np.array([[0, 1, 2], [0, 1, 3]])
        n_nodes, n_elements = write_object_mesh(verts, tris, tmp_path)
        assert (n_nodes, n_elements) == (4, 2)

        from pynumcalc.readers import (
            read_object_mesh_elements, read_object_mesh_nodes,
        )
        np.testing.assert_allclose(read_object_mesh_nodes(tmp_path), verts)
        np.testing.assert_array_equal(read_object_mesh_elements(tmp_path), tris)

    def test_connectivity_is_written_zero_based(self, tmp_path):
        verts = np.zeros((3, 3))
        write_object_mesh(verts, np.array([[0, 1, 2]]), tmp_path)
        row = (tmp_path / "ObjectMesh" / "Elements.txt").read_text().splitlines()[1]
        assert row.split()[:4] == ["0", "0", "1", "2"]
