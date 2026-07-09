"""Per-bin completion accounting for NumCalc runs.

The be.out/ bin listing (or be.npz bin_indices) is the job state — no
separate bookkeeping. Used by `bem status`, resume (`run --missing-only`),
and remote-fetch verification.
"""

from pathlib import Path

import numpy as np

from pynumcalc.sim.readers import _bin_file_ok, list_bins


def available_bins(nc_dir):
    """Sorted 1-based bin indices with results (from be.npz or be.out/).

    A ``be.{i}/`` directory alone does not count: NumCalc leaves an empty or
    header-only ``pBoundary`` stub for bins that crashed or were skipped, so a
    bin is only "available" when its ``pBoundary`` file has real content. This
    matches the loaders (``load_bins_ascii``), so a crashed bin is reported
    missing and ``run --missing-only`` will re-run it.
    """
    nc_dir = Path(nc_dir)
    npz_file = nc_dir / "be.npz"
    if npz_file.exists():
        with np.load(npz_file) as nz:
            return [int(b) for b in nz["bin_indices"]]
    be_out = nc_dir / "be.out"
    if be_out.is_dir():
        return [i for i in list_bins(be_out)
                if _bin_file_ok(be_out / f"be.{i}" / "pBoundary")]
    return []


def missing_bins(nc_dir, nf):
    """1-based bins in [1, nf] without results."""
    return sorted(set(range(1, nf + 1)) - set(available_bins(nc_dir)))


def is_complete(nc_dir, nf):
    """True if every bin in [1, nf] has results."""
    return not missing_bins(nc_dir, nf)


# A bin can be present in be.out/ yet non-converged — NumCalc records that
# only in its log (NC_CommonFunctions.cpp:1167, :1184).
_SOLVER_WARNINGS = (
    "Maximum number of iterations is reached",
    "Iteration method CGS failed",
)


def solver_warnings(nc_dir, logs=None):
    """Solver-health warnings from NC*.out logs, as (log, bin, line) tuples.

    ``bin`` is the 1-based frequency step the warning occurred in, or None
    for warnings before the first step header. These usually indicate a
    defective mesh — re-mesh rather than raising -nitermax.

    Args:
        nc_dir: variant directory
        logs: restrict to these log file names (default: every NC*.out)
    """
    nc_dir = Path(nc_dir)
    files = ([nc_dir / name for name in logs] if logs is not None
             else sorted(nc_dir.glob("NC*.out")))
    found = []
    for log in files:
        if not log.is_file():
            continue
        step = None
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("Step ") and ", Frequency" in line:
                try:
                    step = int(line.split(None, 2)[1].rstrip(","))
                except ValueError:  # garbled/truncated log line
                    pass
            elif any(pat in line for pat in _SOLVER_WARNINGS):
                found.append((log.name, step, line.strip()))
    return found
