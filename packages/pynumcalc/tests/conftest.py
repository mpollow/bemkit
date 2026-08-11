"""Fixtures for the pynumcalc suite (copied from bemkit's: pure numpy)."""

import json
import numpy as np
import pytest
from pathlib import Path




@pytest.fixture()
def tiny_mesh_data():
    """A minimal triangle mesh: tetrahedron (4 vertices, 4 triangles)."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.5, 0.289, 0.816],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ], dtype=np.int32)
    return vertices, triangles


@pytest.fixture()
def numcalc_dir(tmp_path, tiny_mesh_data):
    """Create a minimal numcalc/{variant}/ directory with synthetic data.

    Contains ObjectMesh/, EvalGrid/, NC.inp, be.out/ with 2 freq bins.
    """
    nc = tmp_path / "numcalc" / "test_variant"
    nc.mkdir(parents=True)

    vertices, triangles = tiny_mesh_data
    n_nodes = len(vertices)
    n_elems = len(triangles)

    # ObjectMesh/Nodes.txt
    om = nc / "ObjectMesh"
    om.mkdir()
    with open(om / "Nodes.txt", "w") as f:
        f.write(f"{n_nodes}\n")
        for i, (x, y, z) in enumerate(vertices):
            f.write(f"{i:6d} {x:.15g} {y:.15g} {z:.15g}\n")

    # ObjectMesh/Elements.txt
    with open(om / "Elements.txt", "w") as f:
        f.write(f"{n_elems}\n")
        for i, (a, b, c) in enumerate(triangles):
            f.write(f"{i:6d} {a:6d} {b:6d} {c:6d} 0 0 0\n")

    # EvalGrid — 3 eval points
    eg = nc / "EvalGrid"
    eg.mkdir()
    eval_pts = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    with open(eg / "Nodes.txt", "w") as f:
        f.write("3\n")
        for i, (x, y, z) in enumerate(eval_pts):
            f.write(f"{200000+i:6d} {x:.15g} {y:.15g} {z:.15g}\n")

    # NC.inp with BOUNDARY section
    with open(nc / "NC.inp", "w") as f:
        f.write("TITLE test\n")
        f.write(f"NODES {n_nodes}\n")
        f.write(f"ELEMENTS {n_elems}\n")
        f.write("BOUNDARY\n")
        f.write("ELEM 0 TO 1 VELO 1.0000 -1 0.0 -1\n")
        f.write("RETU\n")

    # be.out with 2 freq bins (be.1, be.2)
    be_out = nc / "be.out"
    for bi in [1, 2]:
        d = be_out / f"be.{bi}"
        d.mkdir(parents=True)
        # pEvalGrid: 3 header lines + 3 data rows
        with open(d / "pEvalGrid", "w") as f:
            f.write("header1\nheader2\nheader3\n")
            for j in range(3):
                re = float(bi * 10 + j)
                im = float(bi * 100 + j)
                f.write(f"{j} {re} {im}\n")
        # pBoundary: 3 header lines + 4 data rows (boundary has 4 nodes)
        with open(d / "pBoundary", "w") as f:
            f.write("header1\nheader2\nheader3\n")
            for j in range(4):
                re = float(bi * 10 + j)
                im = float(bi * 100 + j)
                f.write(f"{j} {re} {im}\n")
        # vEvalGrid: 3 header lines + 3 data rows with 3 components each
        with open(d / "vEvalGrid", "w") as f:
            f.write("header1\nheader2\nheader3\n")
            for j in range(3):
                re = float(bi * 10 + j)
                im = float(bi * 100 + j)
                f.write(f"{j} {re} {im} {re+0.1} {im+0.1} {re+0.2} {im+0.2}\n")
        # vBoundary: 3 header lines + 4 data rows (scalar normal)
        with open(d / "vBoundary", "w") as f:
            f.write("header1\nheader2\nheader3\n")
            for j in range(4):
                re = float(bi * 10 + j)
                im = float(bi * 100 + j)
                f.write(f"{j} {re} {im}\n")

    # eval_offsets.json
    offsets = {
        "sets": {
            "all": {"start": 0, "end": 3, "n_original": 3, "n_valid": 3},
        },
        "dropped": {},
    }
    with open(nc / "eval_offsets.json", "w") as f:
        json.dump(offsets, f)

    return nc
