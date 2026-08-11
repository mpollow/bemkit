import json

import numpy as np
import meshio

from bemkit.config import ModelConfig
from bemkit.sim.boundary import resolve_boundary
from bemkit.sim.export import ExportInfo, export_variant


def _make_model(tmp_path):
    model = tmp_path / "tiny"
    (model / "mesh").mkdir(parents=True)
    (model / "fieldpoints").mkdir()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]], float)
    tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    meshio.write_points_cells(model / "mesh" / "tiny.stl", vertices, [("triangle", tris)])
    np.save(model / "fieldpoints" / "onaxis.npy", np.array([[3, 0, 0], [4, 0, 0]], float))
    (model / "config.toml").write_text(
        '[bem]\nsr = 8000\nnbins = 5\nbem_type = 4\nair_density = 1.2\n'
        'speed_of_sound = 340.0\ntime_factor = -1.0\nis_internal = 0\n'
        '[fieldpoints.onaxis]\nfile = "fieldpoints/onaxis.npy"\n'
        '[BOUNDARY]\ncap = [0, 0, 1.0]\n')
    return model


def test_export_variant_produces_complete_nc_dir(tmp_path):
    config = ModelConfig.from_dir(_make_model(tmp_path))
    info = export_variant(config, "tiny")
    assert isinstance(info, ExportInfo)
    nc = config.model_dir / "numcalc" / "tiny"
    for rel in ("ObjectMesh/Nodes.txt", "ObjectMesh/Elements.txt",
                "EvalGrid/Nodes.txt", "EvalGrid/Elements.txt",
                "NC.inp", "eval_offsets.json"):
        assert (nc / rel).exists(), rel
    assert info.n_nodes == 4 and info.n_elements == 4
    assert info.n_eval_points == 2 and info.fieldpoint_sets == ["onaxis"]
    # config snapshot written with the effective config
    snap = json.loads((nc / "snapshot.json").read_text())
    assert snap["sr"] == 8000 and snap["nbins"] == 5 and snap["bem_type"] == 4


def _make_geometric_model(tmp_path):
    """Model whose BOUNDARY is a geometric near= anchor instead of an index."""
    model = tmp_path / "geo"
    (model / "mesh").mkdir(parents=True)
    (model / "fieldpoints").mkdir()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]], float)
    tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    meshio.write_points_cells(model / "mesh" / "geo.stl", vertices, [("triangle", tris)])
    np.save(model / "fieldpoints" / "onaxis.npy", np.array([[3, 0, 0], [4, 0, 0]], float))
    # Anchor near element 2's centroid.
    centroid2 = vertices[tris[2]].mean(axis=0)
    (model / "config.toml").write_text(
        '[bem]\nsr = 8000\nnbins = 5\nbem_type = 4\nair_density = 1.2\n'
        'speed_of_sound = 340.0\ntime_factor = -1.0\nis_internal = 0\n'
        '[fieldpoints.onaxis]\nfile = "fieldpoints/onaxis.npy"\n'
        f'[BOUNDARY.geo]\ndrv = {{ near = [{centroid2[0]}, {centroid2[1]}, {centroid2[2]}] }}\n')
    return model, 2


def test_export_resolves_geometric_boundary(tmp_path):
    model, expected_elem = _make_geometric_model(tmp_path)
    config = ModelConfig.from_dir(model)
    export_variant(config, "geo")
    nc = config.model_dir / "numcalc" / "geo"
    text = (nc / "NC.inp").read_text()
    assert f"ELEM {expected_elem} TO {expected_elem} VELO 1.0 -1 0.0 -1" in text
    # resolved indices recorded in the snapshot for provenance
    snap = json.loads((nc / "snapshot.json").read_text())
    assert snap["resolved_boundary"] == {
        "drv": [[expected_elem, expected_elem, "VELO", 1.0, 0.0, None]]}


def _make_symmetric_model(tmp_path):
    """Quarter-symmetric model: the mesh is one quarter of the box
    [-1,1] x [-1,1] x [0,1], mirrored by [SYMMETRY] about x = 0 and y = 0."""
    model = tmp_path / "quarter"
    (model / "mesh").mkdir(parents=True)
    (model / "fieldpoints").mkdir()
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], float)
    tris = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [1, 2, 6], [1, 6, 5], [3, 7, 6], [3, 6, 2],
    ])
    meshio.write_points_cells(model / "mesh" / "quarter.stl", vertices,
                              [("triangle", tris)])
    # first point is clear of the object; second is inside the mirrored box
    np.save(model / "fieldpoints" / "grid.npy",
            np.array([[3.0, 3.0, 0.5], [-0.6, -0.3, 0.5]]))
    (model / "config.toml").write_text(
        '[bem]\nsr = 8000\nnbins = 5\nbem_type = 4\nair_density = 1.2\n'
        'speed_of_sound = 340.0\ntime_factor = -1.0\nis_internal = 0\n'
        '[SYMMETRY]\nplanes = [1, 1, 0]\ncoordinates = [0.0, 0.0, 0.0]\n'
        '[fieldpoints.grid]\nfile = "fieldpoints/grid.npy"\n'
        '[BOUNDARY]\ncap = [0, 0, 1.0]\n')
    return model


def test_export_variant_validates_eval_points_against_the_mirrored_mesh(tmp_path):
    """A point inside the mirrored geometry is outside the mesh on disk.
    Keeping it makes NumCalc abort in post-processing, so it must be dropped."""
    config = ModelConfig.from_dir(_make_symmetric_model(tmp_path))
    info = export_variant(config, "quarter", verbose=False)
    assert info.n_eval_points == 1
    nodes = (config.model_dir / "numcalc" / "quarter" / "EvalGrid" / "Nodes.txt")
    assert "-0.6" not in nodes.read_text()
