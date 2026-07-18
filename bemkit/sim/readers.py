"""Raw NumCalc output parsing (be.out ASCII, NC.inp, ObjectMesh/EvalGrid)."""

import numpy as np
from pathlib import Path


def read_eval_grid_nodes(output_dir: str | Path) -> np.ndarray:
    """Read evaluation point coordinates from EvalGrid/Nodes.txt.

    Args:
        output_dir: Directory containing EvalGrid/

    Returns:
        Array of shape (N, 3) with x, y, z coordinates
    """
    nodes_file = Path(output_dir) / "EvalGrid" / "Nodes.txt"
    with open(nodes_file, 'r') as f:
        n_points = int(f.readline().strip())
        coords = []
        for _ in range(n_points):
            parts = f.readline().strip().split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords)


def read_object_mesh_nodes(mesh_dir: str | Path) -> np.ndarray:
    """Read mesh node coordinates from ObjectMesh/Nodes.txt.

    Args:
        mesh_dir: Directory containing ObjectMesh/

    Returns:
        Array of shape (N, 3) with x, y, z coordinates in meters
    """
    nodes_file = Path(mesh_dir) / "ObjectMesh" / "Nodes.txt"
    with open(nodes_file, 'r') as f:
        n_nodes = int(f.readline().strip())
        coords = []
        for _ in range(n_nodes):
            parts = f.readline().strip().split()
            # Format: node_id x y z
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(coords)


def read_object_mesh_elements(mesh_dir: str | Path) -> np.ndarray:
    """Read mesh triangular elements from ObjectMesh/Elements.txt.

    Args:
        mesh_dir: Directory containing ObjectMesh/

    Returns:
        Array of shape (M, 3) with triangle node indices (0-indexed)
    """
    elements_file = Path(mesh_dir) / "ObjectMesh" / "Elements.txt"
    with open(elements_file, 'r') as f:
        n_elements = int(f.readline().strip())
        triangles = []
        for _ in range(n_elements):
            parts = f.readline().strip().split()
            # Format: element_id node1 node2 node3 0 0 0
            triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.array(triangles, dtype=np.int32)


def _read_numcalc_data(filepath: Path) -> np.ndarray:
    """Read complex data from a numcalc output file (pEvalGrid, vEvalGrid, pBoundary, etc.)."""
    data = np.loadtxt(filepath, skiprows=3)
    return data[:, 1] + 1j * data[:, 2]


def _read_numcalc_velocity_data(filepath: Path) -> np.ndarray:
    """Read 3D complex velocity vector from a vEvalGrid file.

    The vEvalGrid format stores one row per point:
    ``index  vx_re  vx_im  vy_re  vy_im  vz_re  vz_im``

    Returns
    -------
    (N, 3) complex128 array, columns = [vx, vy, vz].
    """
    data = np.loadtxt(filepath, skiprows=3)
    return np.stack([
        data[:, 1] + 1j * data[:, 2],
        data[:, 3] + 1j * data[:, 4],
        data[:, 5] + 1j * data[:, 6],
    ], axis=1)


def read_nc_inp_boundary_conditions(nc_inp_file: str | Path) -> dict:
    """Read boundary condition assignments from NC.inp file.

    Parses the BOUNDARY section to extract which elements have VELO (vibrating) BC.

    Args:
        nc_inp_file: Path to NC.inp file

    Returns:
        Dictionary with key "vibrating" mapping to array of element indices (0-indexed).
        E.g., {"vibrating": array([100, 101, 102, 200, 201, ...])}
    """
    vibrating_elements = []

    try:
        with open(nc_inp_file, 'r') as f:
            in_boundary = False
            for line in f:
                line = line.strip()

                # Start of BOUNDARY section
                if line.startswith("BOUNDARY"):
                    in_boundary = True
                    continue

                # End of BOUNDARY section
                if in_boundary and line in ("RETU", "##", ""):
                    break

                # Parse ELEM lines: "ELEM 0 TO 24 VELO 1.0000 -1 0.0 -1"
                # Note: Element indices in NC.inp are 0-indexed directly
                if in_boundary and line.startswith("ELEM"):
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            elem_start = int(parts[1])  # Already 0-indexed
                            # Handle "TO" keyword for ranges
                            if "TO" in parts:
                                to_idx = parts.index("TO")
                                elem_end = int(parts[to_idx + 1])  # inclusive
                                # Check for VELO keyword after range
                                if "VELO" in parts[to_idx + 2:]:
                                    vibrating_elements.extend(range(elem_start, elem_end + 1))
                            else:
                                # Single element: "ELEM 0 VELO ..."
                                if len(parts) >= 3 and parts[2].upper() == "VELO":
                                    vibrating_elements.append(elem_start)
                        except (ValueError, IndexError):
                            # A skipped ELEM line silently changes A_vib (and
                            # thus normalization) — surface it instead.
                            import warnings
                            warnings.warn(
                                f"Malformed ELEM line in {nc_inp_file}: {line!r}",
                                stacklevel=2)
    except FileNotFoundError:
        pass

    return {"vibrating": np.array(vibrating_elements, dtype=np.int32)}


# Known be.out quantities; value = True if 3-vector (one (N,3) row per point)
QUANTITIES = {
    "pBoundary": False,
    "vBoundary": False,
    "pEvalGrid": False,
    "vEvalGrid": True,
}


def list_bins(be_out):
    """Sorted 1-based bin indices present as be.out/be.{i}/ subdirectories."""
    be_out = Path(be_out)
    return sorted(
        int(p.name.split(".")[1])
        for p in be_out.iterdir()
        if p.name.startswith("be.") and p.is_dir()
    )


def _bin_file_ok(f: Path) -> bool:
    # NumCalc leaves empty or header-only stubs (a few bytes) for bins that
    # crashed or were skipped — treat anything <= 10 bytes as absent.
    return f.exists() and f.stat().st_size > 10


def complete_bins(nc_dir):
    """1-based bins in nc_dir/be.out/ whose result files are fully written.

    Within a bin NumCalc writes the boundary files before the eval-grid
    files, so a killed run can leave a complete ``pBoundary`` next to an
    empty ``pEvalGrid``. A bin only counts as complete when every expected
    quantity has content — ``pEvalGrid`` is expected whenever the bundle has
    an ``EvalGrid/``. Resume (``bem run``) and packing rely on this so a
    half-written bin gets recomputed instead of silently kept.
    """
    nc_dir = Path(nc_dir)
    be_out = nc_dir / "be.out"
    if not be_out.is_dir():
        return []
    need_eval = (nc_dir / "EvalGrid" / "Nodes.txt").exists()
    bins = []
    for i in list_bins(be_out):
        d = be_out / f"be.{i}"
        if _bin_file_ok(d / "pBoundary") and (
                not need_eval or _bin_file_ok(d / "pEvalGrid")):
            bins.append(i)
    return bins


def load_bins_ascii(be_out, quantity, nf, bins=None):
    """Load one quantity from be.out ASCII across bins into a single array.

    Args:
        be_out: Path to the be.out/ directory
        quantity: one of QUANTITIES
        nf: total number of frequency bins (array width)
        bins: 1-based bin indices to read (default: all present)

    Returns:
        (N, nf) or (N, nf, 3) complex128 with NaN for missing bins,
        or None if no bin contains the quantity file.
    """
    be_out = Path(be_out)
    vector = QUANTITIES[quantity]
    reader = _read_numcalc_velocity_data if vector else _read_numcalc_data
    if bins is None:
        bins = list_bins(be_out)

    available = [i for i in bins if _bin_file_ok(be_out / f"be.{i}" / quantity)]
    if not available:
        return None

    first = reader(be_out / f"be.{available[0]}" / quantity)
    N = first.shape[0]
    shape = (N, nf, 3) if vector else (N, nf)
    arr = np.full(shape, np.nan + 0j, dtype=np.complex128)
    for i in available:
        if not 1 <= i <= nf:
            continue
        data = first if i == available[0] else reader(be_out / f"be.{i}" / quantity)
        if vector:
            arr[:, i - 1, :] = data
        else:
            arr[:, i - 1] = data
    return arr


def load_bins(nc_dir, quantity, nf):
    """Load one quantity across all bins, preferring packed be.npz.

    Raises FileNotFoundError if neither be.npz nor be.out/ has the quantity.
    """
    nc_dir = Path(nc_dir)
    npz_file = nc_dir / "be.npz"
    if npz_file.exists():
        # be.npz is a complete pack of everything be.out/ contained; if the
        # quantity is missing here, falling back to ASCII would find nothing
        # either, so this raises instead of silently scanning be.out/.
        with np.load(npz_file) as nz:
            if quantity not in nz.files:
                raise FileNotFoundError(f"{quantity} not in {npz_file}")
            return nz[quantity].astype(np.complex128, copy=False)
    arr = load_bins_ascii(nc_dir / "be.out", quantity, nf)
    if arr is None:
        raise FileNotFoundError(f"No {quantity} data in {nc_dir / 'be.out'}")
    return arr
