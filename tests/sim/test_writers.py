"""Tests for bemkit.sim.writers — roundtrip write→read verification."""

import numpy as np
import pytest

from bemkit.sim.writers import (
    write_eval_grid_nodes,
    write_eval_grid_elements,
    write_nc_inp,
    _contiguous_ranges,
    _count_nodes,
    _count_elements,
)
from bemkit.sim.readers import read_eval_grid_nodes


class TestWriteEvalGridNodes:
    def test_roundtrip(self, tmp_path):
        pts = np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        write_eval_grid_nodes(pts, tmp_path)
        read_back = read_eval_grid_nodes(tmp_path)
        np.testing.assert_allclose(read_back, pts)

    def test_creates_directory(self, tmp_path):
        pts = np.array([[0.0, 0.0, 0.0]])
        write_eval_grid_nodes(pts, tmp_path / "sub")
        assert (tmp_path / "sub" / "EvalGrid" / "Nodes.txt").exists()

    def test_node_count_line(self, tmp_path):
        pts = np.zeros((5, 3))
        write_eval_grid_nodes(pts, tmp_path)
        count = _count_nodes(tmp_path / "EvalGrid" / "Nodes.txt")
        assert count == 5

    def test_custom_start_id(self, tmp_path):
        pts = np.array([[1.0, 2.0, 3.0]])
        write_eval_grid_nodes(pts, tmp_path, node_id_start=100)
        # Should still roundtrip correctly (IDs are metadata)
        read_back = read_eval_grid_nodes(tmp_path)
        np.testing.assert_allclose(read_back, pts)


class TestWriteEvalGridElements:
    def test_creates_file(self, tmp_path):
        write_eval_grid_elements(5, tmp_path)
        assert (tmp_path / "EvalGrid" / "Elements.txt").exists()

    def test_element_count(self, tmp_path):
        write_eval_grid_elements(10, tmp_path)
        count = _count_elements(tmp_path / "EvalGrid" / "Elements.txt")
        assert count == 9  # n-1 line segments

    def test_single_point(self, tmp_path):
        write_eval_grid_elements(1, tmp_path)
        count = _count_elements(tmp_path / "EvalGrid" / "Elements.txt")
        assert count == 1  # max(1, 0)

    def test_zero_points_raises(self, tmp_path):
        # A zero-point EvalGrid would write an element referencing an
        # undeclared node — reject instead of producing a corrupt bundle.
        with pytest.raises(ValueError):
            write_eval_grid_elements(0, tmp_path)


class TestContiguousRanges:
    def test_empty(self):
        assert _contiguous_ranges([]) == []

    def test_single_run(self):
        assert _contiguous_ranges([3, 4, 5]) == [(3, 5)]

    def test_gaps_and_order(self):
        assert _contiguous_ranges([7, 1, 2, 9]) == [(1, 2), (7, 7), (9, 9)]


class TestWriteNcInp:
    def test_generates_nc_inp(self, tmp_path):
        # Minimal ObjectMesh/EvalGrid count files (enough elements that the BC
        # indices below are in range).
        for sub, n in [("ObjectMesh", 30), ("EvalGrid", 3)]:
            d = tmp_path / sub
            d.mkdir()
            (d / "Nodes.txt").write_text(f"{n}\n")
            (d / "Elements.txt").write_text(f"{n}\n")
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {"cap": [0, 0, 1.0]},
        }
        write_nc_inp(config, "tiny", tmp_path, vibrating_indices=[1, 2], verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        assert "ELEM 1 TO 2 VELO 1.0000 -1 0.0 -1" in text   # from physical groups
        assert "ELEM 0 TO 0 VELO 1.0000 -1 0.0 -1" in text   # from [BOUNDARY]
        assert "POST PROCESS" in text and text.endswith("END\n")
        assert "{" not in text  # all template fields resolved

    def _counts(self, tmp_path, n_elem=30):
        for sub, n in [("ObjectMesh", n_elem), ("EvalGrid", 3)]:
            d = tmp_path / sub
            d.mkdir()
            (d / "Nodes.txt").write_text(f"{n}\n")
            (d / "Elements.txt").write_text(f"{n}\n")

    def test_variant_boundary_section_used_exclusively(self, tmp_path):
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {
                "left": {"ear": [10, 10, 1.0]},
                "right": {"ear": [20, 20, 1.0]},
            },
        }
        write_nc_inp(config, "left", tmp_path, verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        assert "ELEM 10 TO 10 VELO 1.0000 -1 0.0 -1" in text
        assert "ELEM 20 TO 20" not in text  # other variant's BC excluded

    def test_recipe_suffixed_variant_matches_target_section(self, tmp_path):
        # The config-fallback path must use the same variant matching as
        # export_variant (resolve_boundary): 'left__r05' falls back to
        # [BOUNDARY.left] after stripping the __recipe suffix.
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {"left": {"ear": [10, 10, 1.0]}},
        }
        write_nc_inp(config, "left__r05", tmp_path, verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        assert "ELEM 10 TO 10 VELO 1.0000 -1 0.0 -1" in text

    def test_geometric_anchor_without_mesh_raises(self, tmp_path):
        # near= anchors need a mesh to resolve against; the direct
        # write_nc_inp path has none, so this must raise instead of
        # silently emitting a fully rigid surface.
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {"spot": {"near": [0.0, 0.0, 0.05]}},
        }
        with pytest.raises(ValueError, match="near"):
            write_nc_inp(config, "tiny", tmp_path, verbose=False)

    def test_nested_range_boundary_entries(self, tmp_path):
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {"cap": [[1, 2, 1.0], [5, 6, 0.5]]},
        }
        write_nc_inp(config, "tiny", tmp_path, verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        assert "ELEM 1 TO 2 VELO 1.0000 -1 0.0 -1" in text
        assert "ELEM 5 TO 6 VELO 0.5000 -1 0.0 -1" in text

    def test_out_of_range_boundary_raises(self, tmp_path):
        self._counts(tmp_path, n_elem=5)  # valid indices 0..4
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "BOUNDARY": {"cap": [10, 10, 1.0]},  # 10 >= 5 elements
        }
        with pytest.raises(ValueError, match="out of bounds"):
            write_nc_inp(config, "tiny", tmp_path, verbose=False)

    def test_warns_when_boundary_table_yields_no_elements(self, tmp_path, capsys):
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            # Per-variant table exists but nothing matches variant 'tiny'
            "BOUNDARY": {"left": {"ear": [1, 1, 1.0]}},
        }
        write_nc_inp(config, "tiny", tmp_path, verbose=True)
        assert "fully rigid" in capsys.readouterr().out
        text = (tmp_path / "NC.inp").read_text()
        assert "VELO" not in text  # no BC emitted

    def test_source_sections_written_with_matching_header_count(self, tmp_path):
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "PLANE WAVES": {"1": [1, 0, -1, 0, 1, -1, 0, -1]},
            "POINT SOURCES": {"1": [1, 0, 2, 0, 1, -1, 0, -1]},
        }
        write_nc_inp(config, "tiny", tmp_path, verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        assert "PLANE WAVES\n1 0 -1 0 1 -1 0 -1\n##" in text
        assert "POINT SOURCES\n1 0 2 0 1 -1 0 -1\n##" in text
        # header source counts must match the emitted lines
        assert "1 1 0 0.0000e+00 0 0 0" in text

    def test_malformed_source_entry_raises(self, tmp_path):
        # A silently dropped entry would leave the header count claiming a
        # source line that never appears — NumCalc would misparse the file.
        self._counts(tmp_path)
        config = {
            "modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0,
            "PLANE WAVES": {"1": [1, 0, -1]},  # not 8 values
        }
        with pytest.raises(ValueError, match="PLANE WAVES"):
            write_nc_inp(config, "tiny", tmp_path, verbose=False)

    def test_modelconfig_typed_defaults_fill_omitted_keys(self, tmp_path):
        from bemkit.config import ModelConfig
        self._counts(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.toml").write_text("[bem]\nsr = 8000\nnbins = 5\n")  # sparse
        config = ModelConfig.from_dir(model)
        write_nc_inp(config, "tiny", tmp_path, verbose=False)
        text = (tmp_path / "NC.inp").read_text()
        # time_factor in Main Parameters IV must be the typed default -1.0
        assert " -1.0 " in text
        assert "{" not in text
