"""Tests for numcalckit.readers — all reader functions with synthetic files."""

import numpy as np
import pytest
from pathlib import Path

from numcalckit.readers import (
    read_eval_grid_nodes,
    read_object_mesh_nodes,
    read_object_mesh_elements,
    read_nc_inp_boundary_conditions,
    read_nc_inp_symmetry,
    _read_numcalc_data,
    _read_numcalc_velocity_data,
    QUANTITIES,
    list_bins,
    load_bins,
    load_bins_ascii,
)


class TestReadEvalGridNodes:
    def test_reads_coords(self, numcalc_dir):
        pts = read_eval_grid_nodes(numcalc_dir)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts[0], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(pts[2], [0.0, 0.0, 2.0])

    def test_ignores_node_ids(self, numcalc_dir):
        """Node IDs in file (200000+) should not appear in output."""
        pts = read_eval_grid_nodes(numcalc_dir)
        assert pts.max() <= 2.0


class TestReadObjectMeshNodes:
    def test_reads_nodes(self, numcalc_dir):
        nodes = read_object_mesh_nodes(numcalc_dir)
        assert nodes.shape == (4, 3)
        np.testing.assert_allclose(nodes[0], [0, 0, 0])
        np.testing.assert_allclose(nodes[1], [1, 0, 0])

    def test_node_count(self, numcalc_dir):
        nodes = read_object_mesh_nodes(numcalc_dir)
        assert len(nodes) == 4


class TestReadObjectMeshElements:
    def test_reads_triangles(self, numcalc_dir):
        elems = read_object_mesh_elements(numcalc_dir)
        assert elems.shape == (4, 3)
        assert elems.dtype == np.int32

    def test_zero_indexed(self, numcalc_dir):
        elems = read_object_mesh_elements(numcalc_dir)
        assert elems.min() == 0
        assert elems.max() == 3


class TestReadNcInpBoundaryConditions:
    def test_parses_vibrating_range(self, numcalc_dir):
        bc = read_nc_inp_boundary_conditions(numcalc_dir / "NC.inp")
        vib = bc["vibrating"]
        np.testing.assert_array_equal(vib, [0, 1])

    def test_single_elem(self, tmp_path):
        nc_inp = tmp_path / "NC.inp"
        nc_inp.write_text(
            "BOUNDARY\nELEM 5 TO 5 VELO 1.0 -1 0.0 -1\nRETU\n"
        )
        bc = read_nc_inp_boundary_conditions(nc_inp)
        np.testing.assert_array_equal(bc["vibrating"], [5])

    def test_no_boundary_section(self, tmp_path):
        nc_inp = tmp_path / "NC.inp"
        nc_inp.write_text("TITLE test\nNODES 10\n")
        bc = read_nc_inp_boundary_conditions(nc_inp)
        assert len(bc["vibrating"]) == 0

    def test_missing_file(self, tmp_path):
        bc = read_nc_inp_boundary_conditions(tmp_path / "nope.inp")
        assert len(bc["vibrating"]) == 0

    def test_multiple_ranges(self, tmp_path):
        nc_inp = tmp_path / "NC.inp"
        nc_inp.write_text(
            "BOUNDARY\n"
            "ELEM 0 TO 2 VELO 1.0 -1 0.0 -1\n"
            "ELEM 10 TO 12 VELO 1.0 -1 0.0 -1\n"
            "RETU\n"
        )
        bc = read_nc_inp_boundary_conditions(nc_inp)
        np.testing.assert_array_equal(bc["vibrating"], [0, 1, 2, 10, 11, 12])


class TestReadNcInpSymmetry:
    """The exported bundle, not config.toml, is the authority on what the
    solver actually mirrored — so the viewers read the symmetry back out of
    NC.inp exactly as they read boundary conditions."""

    @staticmethod
    def _write(tmp_path, body):
        nc_inp = tmp_path / "NC.inp"
        nc_inp.write_text(body)
        return nc_inp

    def test_parses_planes_and_coordinates(self, tmp_path):
        nc_inp = self._write(
            tmp_path,
            "ELEMENTS\nObjectMesh/Elements.txt\n##\n"
            "SYMMETRY\n1 1 0\n0.000000 0.000000 0.000000\n##\n"
            "BOUNDARY\n")
        sym = read_nc_inp_symmetry(nc_inp)
        assert sym["planes"] == [1, 1, 0]
        assert sym["coordinates"] == [0.0, 0.0, 0.0]

    def test_keeps_a_nonzero_plane_position(self, tmp_path):
        nc_inp = self._write(
            tmp_path,
            "SYMMETRY\n0 2 0\n0.000000 -0.125000 0.000000\n##\n")
        sym = read_nc_inp_symmetry(nc_inp)
        assert sym["planes"] == [0, 2, 0]
        assert sym["coordinates"] == [0.0, -0.125, 0.0]

    def test_no_symmetry_section_returns_none(self, tmp_path):
        nc_inp = self._write(tmp_path, "NODES\nObjectMesh/Nodes.txt\n##\nBOUNDARY\n")
        assert read_nc_inp_symmetry(nc_inp) is None

    def test_missing_file_returns_none(self, tmp_path):
        """Legacy single-variant layouts have no NC.inp; that is not an error,
        it just means no mirroring — same fallback as the BC reader."""
        assert read_nc_inp_symmetry(tmp_path / "nope.inp") is None

    def test_rejects_a_plane_code_numcalc_would_reject(self, tmp_path):
        nc_inp = self._write(tmp_path, "SYMMETRY\n1 5 0\n0.0 0.0 0.0\n##\n")
        with pytest.raises(ValueError, match="y"):
            read_nc_inp_symmetry(nc_inp)

    def test_rejects_a_negative_plane_code(self, tmp_path):
        nc_inp = self._write(tmp_path, "SYMMETRY\n-1 0 0\n0.0 0.0 0.0\n##\n")
        with pytest.raises(ValueError, match="x"):
            read_nc_inp_symmetry(nc_inp)

    def test_rejects_a_truncated_section(self, tmp_path):
        nc_inp = self._write(tmp_path, "SYMMETRY\n1 1 0\n##\n")
        with pytest.raises(ValueError, match="SYMMETRY"):
            read_nc_inp_symmetry(nc_inp)

    def test_rejects_a_plane_row_that_is_not_three_long(self, tmp_path):
        nc_inp = self._write(tmp_path, "SYMMETRY\n1 1\n0.0 0.0 0.0\n##\n")
        with pytest.raises(ValueError, match="SYMMETRY"):
            read_nc_inp_symmetry(nc_inp)

    def test_reads_a_real_exported_bundle(self, numcalc_dir):
        """The fixture bundle has no SYMMETRY section — the common case."""
        assert read_nc_inp_symmetry(numcalc_dir / "NC.inp") is None


class TestReadNumcalcData:
    def test_reads_complex(self, numcalc_dir):
        data = _read_numcalc_data(numcalc_dir / "be.out" / "be.1" / "pEvalGrid")
        assert data.dtype == np.complex128
        assert len(data) == 3
        # be.1: re = 10+j, im = 100+j for j=0,1,2
        assert data[0] == pytest.approx(10 + 100j)
        assert data[2] == pytest.approx(12 + 102j)

    def test_reads_second_bin(self, numcalc_dir):
        data = _read_numcalc_data(numcalc_dir / "be.out" / "be.2" / "pEvalGrid")
        assert data[0] == pytest.approx(20 + 200j)


class TestReadNumcalcVelocityData:
    def test_reads_velocity_vectors(self, tmp_path):
        d = tmp_path / "be.out" / "be.1"
        d.mkdir(parents=True)
        with open(d / "vEvalGrid", "w") as f:
            f.write("h1\nh2\nh3\n")
            # index vx_re vx_im vy_re vy_im vz_re vz_im
            f.write("0 1.0 2.0 3.0 4.0 5.0 6.0\n")
            f.write("1 7.0 8.0 9.0 10.0 11.0 12.0\n")
        vel = _read_numcalc_velocity_data(d / "vEvalGrid")
        assert vel.shape == (2, 3)
        assert vel[0, 0] == pytest.approx(1 + 2j)
        assert vel[0, 2] == pytest.approx(5 + 6j)
        assert vel[1, 1] == pytest.approx(9 + 10j)


class TestLoadBins:
    def test_list_bins(self, numcalc_dir):
        bins = list_bins(numcalc_dir / "be.out")
        assert bins == sorted(bins)
        assert all(isinstance(b, int) and b >= 1 for b in bins)

    def test_scalar_quantity_shape(self, numcalc_dir):
        nf = 4  # deliberately > the fixture's 2 bins — exercises NaN padding
        p = load_bins(numcalc_dir, "pEvalGrid", nf)
        assert p.ndim == 2 and p.shape[1] == nf
        assert p.dtype == np.complex128

    def test_vector_quantity_shape(self, numcalc_dir):
        nf = 4
        v = load_bins(numcalc_dir, "vEvalGrid", nf)
        assert v.ndim == 3 and v.shape[1:] == (nf, 3)

    def test_missing_bins_are_nan(self, numcalc_dir):
        import shutil
        shutil.rmtree(numcalc_dir / "be.out" / "be.2")
        p = load_bins(numcalc_dir, "pEvalGrid", 4)
        assert np.all(np.isnan(p[:, 1]))
        assert not np.any(np.isnan(p[:, 0]))

    def test_ascii_returns_none_without_data(self, tmp_path):
        (tmp_path / "be.out" / "be.1").mkdir(parents=True)
        assert load_bins_ascii(tmp_path / "be.out", "pEvalGrid", 4) is None

    def test_load_bins_raises_without_source(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bins(tmp_path, "pEvalGrid", 4)
