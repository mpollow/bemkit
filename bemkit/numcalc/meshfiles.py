"""The ``ObjectMesh/`` and ``EvalGrid/`` node and element files NumCalc reads.

These take **arrays**, never mesh-file paths: reading a ``.msh``/``.stl`` needs
meshio and bemkit's millimetre/metre convention, both of which are bemkit
concerns. Callers convert first (``bemkit.meshing.io.read_mesh_triangles``) and
pass coordinates in metres, which is what keeps this module free of any bemkit
import.
"""

from pathlib import Path

import numpy as np

#: Eval nodes are numbered from here to keep them clear of object-mesh ids.
EVAL_NODE_ID_START = 200000


def write_eval_grid_nodes(points, output_dir, node_id_start=EVAL_NODE_ID_START):
    """Write evaluation points to ``EvalGrid/Nodes.txt``.

    Args:
        points: (N, 3) coordinates in metres
        output_dir: directory containing ``EvalGrid/``
        node_id_start: first node id
    """
    points = np.asarray(points, dtype=float)
    d = Path(output_dir) / "EvalGrid"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "Nodes.txt", "w") as f:
        f.write(f"{len(points)}\n")
        f.writelines(f"{node_id_start + i:6d} {x:.15g} {y:.15g} {z:.15g}\n"
                     for i, (x, y, z) in enumerate(points))


def write_eval_grid_elements(n_points, output_dir,
                             node_id_start=EVAL_NODE_ID_START):
    """Write ``EvalGrid/Elements.txt`` — line elements joining consecutive points.

    NumCalc needs elements for the evaluation mesh even though only the nodes
    carry results, so consecutive points are strung together.
    """
    if n_points < 1:
        raise ValueError(
            f"write_eval_grid_elements requires at least 1 eval point, got "
            f"{n_points} (a zero-point EvalGrid would reference undeclared nodes)")
    d = Path(output_dir) / "EvalGrid"
    d.mkdir(parents=True, exist_ok=True)

    n_elements = max(1, n_points - 1)

    def rows():
        for i in range(n_elements):
            n1 = node_id_start + i
            if i < n_points - 1:
                n2 = node_id_start + i + 1
                n3 = node_id_start + i + 2 if i < n_points - 2 else n2
            else:
                n2 = n3 = n1
            yield f"{node_id_start + i:6d} {n1:6d} {n2:6d} {n3:6d} 2 0 1\n"

    with open(d / "Elements.txt", "w") as f:
        f.write(f"{n_elements}\n")
        f.writelines(rows())


def write_object_mesh(vertices, triangles, output_dir):
    """Write ``ObjectMesh/Nodes.txt`` and ``Elements.txt``.

    Args:
        vertices: (N, 3) coordinates **in metres**
        triangles: (M, 3) 0-based connectivity
        output_dir: directory containing ``ObjectMesh/``

    Returns:
        ``(n_nodes, n_elements)``
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    d = Path(output_dir) / "ObjectMesh"
    d.mkdir(parents=True, exist_ok=True)

    # Streamed, not built into a list first: these are the only files whose
    # length scales with the mesh, and BEM peak memory is the binding
    # constraint on a run. Materialising the lines cost ~100 MB of Python heap
    # for a 500k-element mesh, for a 17 MB file.
    with open(d / "Nodes.txt", "w") as f:
        f.write(f"{len(vertices)}\n")
        f.writelines(f"{i:6d} {x:.15g} {y:.15g} {z:.15g}\n"
                     for i, (x, y, z) in enumerate(vertices))

    with open(d / "Elements.txt", "w") as f:
        f.write(f"{len(triangles)}\n")
        f.writelines(f"{i:6d} {a:6d} {b:6d} {c:6d} 0 0 0\n"
                     for i, (a, b, c) in enumerate(triangles))

    return len(vertices), len(triangles)
