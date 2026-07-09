"""Functions for writing BEM data in numcalc (Mesh2HRTF) format."""

from datetime import datetime
from importlib.resources import files as _resource_files
from pathlib import Path

import numpy as np

from pynumcalc.meshing.io import read_mesh_triangles


def write_eval_grid_nodes(points: np.ndarray, output_dir: str | Path, node_id_start: int = 200000) -> None:
    """Write evaluation points to EvalGrid/Nodes.txt format.

    Args:
        points: Array of shape (N, 3) with x, y, z coordinates
        output_dir: Directory where EvalGrid/ will be created
        node_id_start: Starting node ID (default 200000)
    """
    output_dir = Path(output_dir)
    eval_grid_dir = output_dir / "EvalGrid"
    eval_grid_dir.mkdir(parents=True, exist_ok=True)

    nodes_file = eval_grid_dir / "Nodes.txt"
    n_points = len(points)

    with open(nodes_file, 'w') as f:
        # First line: total number of nodes (no padding)
        f.write(f"{n_points}\n")

        # Write each node: node_id x y z (coordinates assumed in meters)
        for i, (x, y, z) in enumerate(points):
            node_id = node_id_start + i
            f.write(f"{node_id:6d} {x:.15g} {y:.15g} {z:.15g}\n")


def write_eval_grid_elements(n_points: int, output_dir: str | Path, node_id_start: int = 200000) -> None:
    """Write evaluation grid elements to EvalGrid/Elements.txt format.

    Creates line elements connecting consecutive evaluation points.

    Args:
        n_points: Number of evaluation points
        output_dir: Directory where EvalGrid/ will be created
        node_id_start: Starting node ID (default 200000)
    """
    if n_points < 1:
        raise ValueError(
            f"write_eval_grid_elements requires at least 1 eval point, got {n_points} "
            f"(a zero-point EvalGrid would reference undeclared nodes)")
    output_dir = Path(output_dir)
    eval_grid_dir = output_dir / "EvalGrid"
    eval_grid_dir.mkdir(parents=True, exist_ok=True)

    elements_file = eval_grid_dir / "Elements.txt"
    n_elements = max(1, n_points - 1)  # Create n-1 line segments (or 1 if only 1 point)

    with open(elements_file, 'w') as f:
        # First line: total number of elements (no padding)
        f.write(f"{n_elements}\n")

        # Write line elements: element_id node1 node2 node3 type 0 1
        for i in range(n_elements):
            element_id = node_id_start + i
            node1 = node_id_start + i

            if i < n_points - 1:
                # Normal case: connect consecutive points
                node2 = node_id_start + i + 1
                node3 = node_id_start + i + 2 if i < n_points - 2 else node2
            else:
                # Last element (or single point case)
                node2 = node1
                node3 = node1

            f.write(f"{element_id:6d} {node1:6d} {node2:6d} {node3:6d} 2 0 1\n")


def write_object_mesh_nodes(mesh_file: str | Path, output_dir: str | Path) -> int:
    """Write mesh nodes to ObjectMesh/Nodes.txt format.

    Reads nodes from a mesh file and writes to numcalc format in meters.

    Args:
        mesh_file: Path to mesh file (.msh, .stl, .ply, etc.)
        output_dir: Directory where ObjectMesh/ will be created

    Returns:
        Number of nodes written
    """
    mesh_file = Path(mesh_file)
    output_dir = Path(output_dir)
    object_mesh_dir = output_dir / "ObjectMesh"
    object_mesh_dir.mkdir(parents=True, exist_ok=True)

    vertices, _ = read_mesh_triangles(mesh_file)

    nodes_file = object_mesh_dir / "Nodes.txt"
    n_nodes = len(vertices)

    with open(nodes_file, 'w') as f:
        f.write(f"{n_nodes}\n")

        for node_id, (x, y, z) in enumerate(vertices):
            f.write(f"{node_id:6d} {x:.15g} {y:.15g} {z:.15g}\n")

    return n_nodes


def write_object_mesh_elements(mesh_file: str | Path, output_dir: str | Path) -> int:
    """Write mesh elements to ObjectMesh/Elements.txt format.

    Reads triangular elements from a mesh file and writes to numcalc format.

    Args:
        mesh_file: Path to mesh file (.msh, .stl, .ply, etc.)
        output_dir: Directory where ObjectMesh/ will be created

    Returns:
        Number of elements written
    """
    mesh_file = Path(mesh_file)
    output_dir = Path(output_dir)
    object_mesh_dir = output_dir / "ObjectMesh"
    object_mesh_dir.mkdir(parents=True, exist_ok=True)

    _, triangles = read_mesh_triangles(mesh_file)

    elements_file = object_mesh_dir / "Elements.txt"
    n_elements = len(triangles)

    with open(elements_file, 'w') as f:
        f.write(f"{n_elements}\n")

        # meshio returns 0-indexed connectivity — write directly
        for element_id, (n1, n2, n3) in enumerate(triangles):
            f.write(f"{element_id:6d} {n1:6d} {n2:6d} {n3:6d} 0 0 0\n")

    return n_elements


def _count_nodes(nodes_file: str | Path) -> int:
    """Read node count from first line of Nodes.txt file."""
    with open(nodes_file, 'r') as f:
        return int(f.readline().strip())


def _count_elements(elements_file: str | Path) -> int:
    """Read element count from first line of Elements.txt file."""
    with open(elements_file, 'r') as f:
        return int(f.readline().strip())


def _contiguous_ranges(indices):
    """Collapse sorted integer indices into inclusive (start, end) runs."""
    ranges = []
    for idx in sorted(int(i) for i in indices):
        if ranges and idx == ranges[-1][1] + 1:
            ranges[-1][1] = idx
        else:
            ranges.append([idx, idx])
    return [tuple(r) for r in ranges]


def _boundary_ranges_from_config(bc, variant):
    """Flatten a config [BOUNDARY] table into (start, end, velocity) tuples.

    A per-variant subsection ``[BOUNDARY.<variant>]`` is used exclusively when
    present; otherwise the global entries apply. Each entry value is either a
    single ``[s, e, v]`` range or a list of such ranges.
    """
    if variant in bc and isinstance(bc[variant], dict):
        entries = bc[variant]
    else:
        entries = bc
    ranges = []
    for val in entries.values():
        if not isinstance(val, list) or len(val) < 1:
            continue
        if isinstance(val[0], list):        # list of [s, e, v] ranges
            ranges += [(r[0], r[1], r[2]) for r in val if len(r) >= 3]
        elif len(val) >= 3:                 # single [s, e, v]
            ranges.append((val[0], val[1], val[2]))
    return ranges


def write_nc_inp(config, variant, nc_dir, vibrating_indices=None,
                 boundary_ranges=None, verbose=True):
    """Generate nc_dir/NC.inp from the packaged template.

    Args:
        config: ModelConfig or raw vars.toml dict (needs modelname, sr, nbins,
            bem_type, air_density, speed_of_sound; optional time_factor,
            is_internal, SYMMETRY, BOUNDARY, PLANE WAVES, POINT SOURCES, velocity)
        variant: variant name. When config has a [BOUNDARY.<variant>] dict
            subsection, it is used EXCLUSIVELY; otherwise the global
            [BOUNDARY] entries apply.
        nc_dir: numcalc/{variant}/ directory; ObjectMesh/ and EvalGrid/ counts
            must already be written
        vibrating_indices: element indices with VELO BC (from mesh physical
            groups); [BOUNDARY] entries are appended additionally
        boundary_ranges: pre-resolved (start, end, velocity) tuples to emit
            instead of parsing config[BOUNDARY] (used by export_variant, which
            resolves geometric ``near=`` anchors against the mesh first). When
            None, the config [BOUNDARY] table is parsed here (literal forms only).
    """
    nc_dir = Path(nc_dir)
    template = _resource_files("pynumcalc.sim").joinpath("templates/NC.inp").read_text()

    n_points = _count_nodes(nc_dir / "ObjectMesh" / "Nodes.txt")
    n_tri = _count_elements(nc_dir / "ObjectMesh" / "Elements.txt")
    n_eval = _count_nodes(nc_dir / "EvalGrid" / "Nodes.txt")
    n_evalelem = _count_elements(nc_dir / "EvalGrid" / "Elements.txt")

    vibrating_ranges = _contiguous_ranges(
        vibrating_indices if vibrating_indices is not None else [])

    v = dict(config.raw) if hasattr(config, "raw") else dict(config)
    if hasattr(config, "__dataclass_fields__"):
        # ModelConfig: let its typed field defaults fill keys vars.toml omitted,
        # so dict-style and attribute access never diverge.
        for name in ("modelname", "sr", "nbins", "bem_type", "air_density",
                     "speed_of_sound", "time_factor", "is_internal"):
            v.setdefault(name, getattr(config, name))
    v.update(
        n_total_nodes=n_points + n_eval,
        n_total_elements=n_tri + n_evalelem,
        date=datetime.today().strftime("%Y-%m-%d"),
        n_frequencies=v["nbins"] - 1,
        freq_time=(v["nbins"] - 1) * 1e-6,
        max_frequency=v["sr"] / 2,
        n_planewaves=len(v.get("PLANE WAVES", {})),
        n_pointsources=len(v.get("POINT SOURCES", {})),
    )
    v.setdefault("time_factor", 1.0)
    v.setdefault("is_internal", 0)

    sym = v.get("SYMMETRY", {})
    v["num_symmetric_planes"] = sum(1 for p in sym.get("planes", []) if p != 0)

    if verbose:
        print(f"Mesh: {n_points} nodes, {n_tri} elements | EvalGrid: {n_eval} nodes")

    with open(nc_dir / "NC.inp", "w") as nc:
        for line in template.splitlines(keepends=True):
            nc.write(line.format(**v))

        nc.write('NODES\nObjectMesh/Nodes.txt\nEvalGrid/Nodes.txt\n##\n')
        nc.write('ELEMENTS\nObjectMesh/Elements.txt\nEvalGrid/Elements.txt\n##\n')

        if v["num_symmetric_planes"] > 0:
            p = sym.get("planes", [0, 0, 0])
            c = sym.get("coordinates", [0.0, 0.0, 0.0])
            nc.write(f'SYMMETRY\n{p[0]} {p[1]} {p[2]}\n'
                     f'{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n##\n')

        nc.write('BOUNDARY\n')
        # Assemble every VELO line as (start, end, velocity) tuples first, so
        # they can be validated against the element count and so an empty
        # result (a fully-rigid surface) can be flagged.
        vel = v.get("velocity", 1.0)
        elems = [(s, e, vel) for s, e in vibrating_ranges]
        had_boundary_table = "BOUNDARY" in v
        if boundary_ranges is not None:
            elems += list(boundary_ranges)
        elif had_boundary_table:
            elems += _boundary_ranges_from_config(v["BOUNDARY"], variant)

        for s, e, v0 in elems:
            if not (0 <= int(s) <= int(e) < n_tri):
                raise ValueError(
                    f"BOUNDARY element range [{s}, {e}] for variant '{variant}' is "
                    f"out of bounds: the mesh has {n_tri} elements (valid indices "
                    f"0..{n_tri - 1}). A stale index usually means the BC was picked "
                    f"on a different mesh — re-pick or use geometric 'near=' anchors.")
            nc.write(f'ELEM {int(s)} TO {int(e)} VELO {v0:.4f} -1 0.0 -1\n')

        if not elems and (had_boundary_table or vibrating_indices):
            print(f"  WARNING: variant '{variant}' has a BOUNDARY definition but no "
                  f"vibrating elements were emitted — the surface will be fully rigid. "
                  f"Check that [BOUNDARY.{variant}] exists and its entries are well-formed.")
        nc.write('RETU\n##\n')

        for section in ('PLANE WAVES', 'POINT SOURCES'):
            if section in v:
                nc.write(f'{section}\n')
                for val in v[section].values():
                    if isinstance(val, list) and len(val) == 8:
                        nc.write(f'{val[0]} {val[1]} {val[2]} {val[3]} '
                                 f'{val[4]} -1 {val[6]} -1\n')
                nc.write('##\n')

        nc.write('POST PROCESS\n##\nEND\n')

    if verbose:
        print(f"NC.inp → {nc_dir / 'NC.inp'}")
