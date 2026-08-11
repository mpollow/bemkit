"""Per-bin completion accounting for NumCalc runs.

The be.out/ bin listing (or be.npz bin_indices) is the job state — no
separate bookkeeping. Used by `bem status`, resume (`run --missing-only`),
and remote-fetch verification.
"""

from pathlib import Path

import numpy as np

from pynumcalc.readers import complete_bins


def available_bins(nc_dir):
    """Sorted 1-based bin indices with results (from be.npz or be.out/).

    A ``be.{i}/`` directory alone does not count: NumCalc leaves empty or
    header-only stubs for bins that crashed, were skipped, or were killed
    mid-write (boundary files are written before eval files), so a bin is
    only "available" when every expected quantity file has real content
    (see ``readers.complete_bins``). A crashed or half-written bin is thus
    reported missing and ``bem run`` will re-run it.
    """
    nc_dir = Path(nc_dir)
    npz_file = nc_dir / "be.npz"
    if npz_file.exists():
        with np.load(npz_file) as nz:
            return [int(b) for b in nz["bin_indices"]]
    return complete_bins(nc_dir)


def missing_bins(nc_dir, nf):
    """1-based bins in [1, nf] without results."""
    return sorted(set(range(1, nf + 1)) - set(available_bins(nc_dir)))


def is_complete(nc_dir, nf):
    """True if every bin in [1, nf] has results."""
    return not missing_bins(nc_dir, nf)


def _numcalc_procs():
    """Yield ``(pid, cwd)`` for every running NumCalc process.

    NumCalc always runs with a variant directory as its working directory
    (``LocalRunner``, or a manual ``cd numcalc/<v> && NumCalc``), so the cwd
    of the process is the whole of its identity — model and variant both.
    Best effort: yields nothing on a platform without ``/proc``, and skips a
    process that vanished mid-scan or belongs to another user.
    """
    import os
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() != "NumCalc":
                continue
            cwd = Path(os.readlink(p / "cwd"))
        except OSError:  # process vanished or belongs to another user
            continue
        yield int(p.name), cwd


def numcalc_pids(nc_dir):
    """PIDs of NumCalc processes currently running with CWD = nc_dir.

    State is derived, not tracked (see ``_numcalc_procs``). Returns [] on
    platforms without /proc.
    """
    nc_dir = Path(nc_dir).resolve()
    return sorted(pid for pid, cwd in _numcalc_procs() if cwd == nc_dir)


def scan_numcalc():
    """Every running NumCalc job as ``{variant_dir: [pid, ...]}``.

    The machine-wide counterpart of ``numcalc_pids`` — it does not filter by
    directory, so one call sees jobs across every model. ``bem ps`` turns each
    key into a model / variant / progress line.
    """
    jobs = {}
    for pid, cwd in _numcalc_procs():
        jobs.setdefault(cwd, []).append(pid)
    return {d: sorted(pids) for d, pids in jobs.items()}


def running_variants(model_dir, variants):
    """Map each variant with a live NumCalc to its PIDs (absent when idle).

    Both `run` and `pack` need this: a second NumCalc in a variant directory
    races the first one's be.out/ writes, and packing deletes be.out/ from
    under a solver still writing into it.

    Args:
        model_dir: model directory (variants live in ``numcalc/<name>/``)
        variants: variant names to check

    Returns:
        ``{variant: [pid, ...]}``, containing only the busy variants.
    """
    model_dir = Path(model_dir)
    busy = {}
    for v in variants:
        pids = numcalc_pids(model_dir / "numcalc" / v)
        if pids:
            busy[v] = pids
    return busy


def current_step(nc_dir):
    """1-based bin the newest NC*.out log last reported working on, or None."""
    nc_dir = Path(nc_dir)
    logs = sorted(nc_dir.glob("NC*.out"), key=lambda f: f.stat().st_mtime)
    if not logs:
        return None
    step = None
    for line in logs[-1].read_text(errors="replace").splitlines():
        if line.startswith("Step ") and ", Frequency" in line:
            try:
                step = int(line.split(None, 2)[1].rstrip(","))
            except ValueError:
                pass
    return step


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
