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
    lines = [f"{len(points)}"]
    lines += [f"{node_id_start + i:6d} {x:.15g} {y:.15g} {z:.15g}"
              for i, (x, y, z) in enumerate(points)]
    (d / "Nodes.txt").write_text("\n".join(lines) + "\n")


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
    lines = [f"{n_elements}"]
    for i in range(n_elements):
        n1 = node_id_start + i
        if i < n_points - 1:
            n2 = node_id_start + i + 1
            n3 = node_id_start + i + 2 if i < n_points - 2 else n2
        else:
            n2 = n3 = n1
        lines.append(f"{node_id_start + i:6d} {n1:6d} {n2:6d} {n3:6d} 2 0 1")
    (d / "Elements.txt").write_text("\n".join(lines) + "\n")


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

    nodes = [f"{len(vertices)}"]
    nodes += [f"{i:6d} {x:.15g} {y:.15g} {z:.15g}"
              for i, (x, y, z) in enumerate(vertices)]
    (d / "Nodes.txt").write_text("\n".join(nodes) + "\n")

    elems = [f"{len(triangles)}"]
    elems += [f"{i:6d} {a:6d} {b:6d} {c:6d} 0 0 0"
              for i, (a, b, c) in enumerate(triangles)]
    (d / "Elements.txt").write_text("\n".join(elems) + "\n")

    return len(vertices), len(triangles)
