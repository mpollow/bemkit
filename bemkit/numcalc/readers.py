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


def _load_numcalc_rows(filepath: Path) -> np.ndarray:
    """np.loadtxt past the 3-line header, raising FileNotFoundError if the
    file isn't completely written yet.

    NumCalc writes rows incrementally, so a bin actively being computed can
    have a file that is nonempty but short a few (or most) rows -- reading
    it mid-write would silently return a truncated array instead of failing
    loudly, and callers that expect one row per mesh element/eval point
    would then either crash on a length mismatch or misalign data. The
    third header line declares the row count NumCalc intends to write
    (confirmed against this project's own output: e.g. ``0 35778`` for a
    35778-element ObjectMesh, ``1  1730`` for 1730 eval points), so compare
    against that instead of guessing from file size. Callers already treat
    FileNotFoundError as "not computed yet" (see e.g. results_viewer's
    frequency slider), so raise that instead of letting a crash or a
    silently-truncated read surface.
    """
    with open(filepath) as f:
        f.readline()  # "Mesh2HRTF 1.0.0"
        f.readline()  # column-count line
        header3 = f.readline()
    try:
        expected_rows = int(header3.split()[-1])
    except (IndexError, ValueError):
        expected_rows = None

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "input contained no data"
        data = np.loadtxt(filepath, skiprows=3)
    # loadtxt collapses a single data row to 1-D (ncols,) instead of
    # (1, ncols) -- normalize so a genuinely-complete one-row file isn't
    # miscounted as empty, and downstream column-indexing doesn't break.
    if data.size:
        data = np.atleast_2d(data)

    n_rows = data.shape[0] if data.ndim == 2 else 0
    if n_rows == 0 or (expected_rows is not None and n_rows < expected_rows):
        raise FileNotFoundError(
            f"{filepath}: incomplete ({n_rows}/"
            f"{expected_rows if expected_rows is not None else '?'} rows "
            f"-- still being written?)")
    return data


def _read_numcalc_data(filepath: Path) -> np.ndarray:
    """Read complex data from a numcalc output file (pEvalGrid, vEvalGrid, pBoundary, etc.)."""
    data = _load_numcalc_rows(filepath)
    return data[:, 1] + 1j * data[:, 2]


def _read_numcalc_velocity_data(filepath: Path) -> np.ndarray:
    """Read 3D complex velocity vector from a vEvalGrid file.

    The vEvalGrid format stores one row per point:
    ``index  vx_re  vx_im  vy_re  vy_im  vz_re  vz_im``

    Returns
    -------
    (N, 3) complex128 array, columns = [vx, vy, vz].
    """
    data = _load_numcalc_rows(filepath)
    return np.stack([
        data[:, 1] + 1j * data[:, 2],
        data[:, 3] + 1j * data[:, 4],
        data[:, 5] + 1j * data[:, 6],
    ], axis=1)


def read_nc_inp_boundary_conditions(nc_inp_file: str | Path) -> dict:
    """Read the driven (VELO) element indices from an NC.inp file.

    Deliberately a *section scanner*, not a deck parser: this feeds ``A_vib``
    and therefore every ``norm="acc"`` result, so it must keep working on a
    partial or hand-edited bundle that ``Deck.read`` would reject. A missing
    file or a missing BOUNDARY section means "nothing driven", not an error.

    The individual ``ELEM`` lines are handed to
    :func:`bemkit.numcalc.deck.parse_bc_line`, so the ranges are extracted by
    the same code that reads a complete deck.

    Args:
        nc_inp_file: Path to NC.inp file

    Returns:
        Dictionary with key "vibrating" mapping to array of element indices (0-indexed).
        E.g., {"vibrating": array([100, 101, 102, 200, 201, ...])}
    """
    from bemkit.numcalc.deck import parse_bc_line

    vibrating_elements = []
    try:
        with open(nc_inp_file, 'r') as f:
            in_boundary = False
            for line in f:
                line = line.strip()
                if line.startswith("BOUNDARY"):
                    in_boundary = True
                    continue
                if in_boundary and line in ("RETU", "##", ""):
                    break
                if in_boundary and line.startswith("ELEM"):
                    try:
                        bc = parse_bc_line(line.split(),
                                           where=f"{nc_inp_file}: ")
                    except ValueError as exc:
                        # A skipped ELEM line silently changes A_vib (and
                        # thus normalization) — surface it instead.
                        import warnings
                        warnings.warn(str(exc), stacklevel=2)
                        continue
                    if bc.kind == "VELO":
                        vibrating_elements.extend(range(bc.start, bc.end + 1))
    except FileNotFoundError:
        pass

    return {"vibrating": np.array(vibrating_elements, dtype=np.int32)}


def read_nc_inp_symmetry(nc_inp_file: str | Path) -> dict | None:
    """Read the SYMMETRY section from an NC.inp file.

    The variant directory is a self-contained job bundle, so a viewer that
    wants to draw what the solver actually saw must read the symmetry back
    from the bundle rather than from config.toml, which may have moved on.

    Args:
        nc_inp_file: Path to NC.inp

    Returns:
        ``{"planes": [px, py, pz], "coordinates": [cx, cy, cz]}``, or None
        when the file has no SYMMETRY section (or does not exist — legacy
        single-variant layouts, treated as "no symmetry" like the boundary
        condition reader treats a missing file).

    Raises:
        ValueError: on a malformed or truncated section, or a plane code
            outside 0-4. NumCalc rejects the same codes
            (``NC_Input.cpp:489``), so a bundle carrying one could never have
            produced results.
    """
    try:
        with open(nc_inp_file, 'r') as f:
            lines = [ln.strip() for ln in f]
    except FileNotFoundError:
        return None

    for i, line in enumerate(lines):
        if line.startswith("SYMMETRY"):
            break
    else:
        return None

    rows = [ln for ln in lines[i + 1:i + 3] if ln and not ln.startswith("##")]
    if len(rows) < 2:
        raise ValueError(
            f"Truncated SYMMETRY section in {nc_inp_file}: expected a plane-code "
            f"row and a coordinate row, got {len(rows)}")

    try:
        planes = [int(v) for v in rows[0].split()]
        coordinates = [float(v) for v in rows[1].split()]
    except ValueError as exc:
        raise ValueError(
            f"Malformed SYMMETRY section in {nc_inp_file}: {exc}") from exc

    if len(planes) != 3 or len(coordinates) != 3:
        raise ValueError(
            f"Malformed SYMMETRY section in {nc_inp_file}: need three plane "
            f"codes and three coordinates, got {len(planes)} and "
            f"{len(coordinates)}")

    for axis, code in zip("xyz", planes):
        if not 0 <= code <= 4:
            raise ValueError(
                f"Invalid SYMMETRY plane code {code} for axis {axis} in "
                f"{nc_inp_file}: NumCalc accepts 0-4 only")

    return {"planes": planes, "coordinates": coordinates}


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

    candidates = [i for i in bins if _bin_file_ok(be_out / f"be.{i}" / quantity)]
    if not candidates:
        return None

    arr = None
    for i in candidates:
        if not 1 <= i <= nf:
            continue
        try:
            data = reader(be_out / f"be.{i}" / quantity)
        except FileNotFoundError:
            # Passed the cheap size check but still has no data rows --
            # e.g. the currently-computing bin, whose header NumCalc
            # flushes before any result rows. Treat as missing, same as a
            # bin that failed the size check outright.
            continue
        if arr is None:
            N = data.shape[0]
            shape = (N, nf, 3) if vector else (N, nf)
            arr = np.full(shape, np.nan + 0j, dtype=np.complex128)
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
