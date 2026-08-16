"""Per-bin completion accounting and solver health for NumCalc runs.

The be.out/ bin listing (or be.npz bin_indices) is the job state — no
separate bookkeeping. Used by `bem status`, resume (`run --missing-only`),
and remote-fetch verification.

Presence is not quality, though, so this module also reads the NC*.out logs:
``solver_warnings`` for the two things NumCalc calls out by name, and
``convergence`` for the per-bin numbers it reports on every step — how many
CGS iterations a bin needed and what residual it reached.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from numcalckit.readers import complete_bins


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


def _logs_by_age(nc_dir, logs=None):
    """NC*.out files, oldest first — or the named subset, in the order given.

    Age order is what makes "last one wins" mean "the run that actually
    produced the be.out/ on disk": a resume writes a fresh log covering bins
    an earlier one already reported on, so one variant directory routinely
    holds several logs with overlapping step ranges.
    """
    nc_dir = Path(nc_dir)
    if logs is not None:
        return [p for p in (nc_dir / name for name in logs) if p.is_file()]
    return sorted(nc_dir.glob("NC*.out"), key=lambda f: f.stat().st_mtime)


def _step_header(line):
    """``(bin, frequency_hz)`` for a ``Step N, Frequency = F Hz`` line, else None.

    ``None`` covers both "not a step header" and "a step header too garbled to
    read a bin out of" — a truncated log ends mid-line often enough that this
    must not raise. The frequency alone may be unreadable while the bin is
    fine, in which case it comes back None and the bin still counts.
    """
    if not (line.startswith("Step ") and ", Frequency" in line):
        return None
    try:
        step = int(line.split(None, 2)[1].rstrip(","))
    except ValueError:  # garbled/truncated log line
        return None
    try:
        freq = float(line.split("=", 1)[1].split()[0])
    except (IndexError, ValueError):
        freq = None
    return step, freq


def current_step(nc_dir):
    """1-based bin the newest NC*.out log last reported working on, or None."""
    logs = _logs_by_age(nc_dir)
    if not logs:
        return None
    step = None
    for line in logs[-1].read_text(errors="replace").splitlines():
        header = _step_header(line)
        if header is not None:
            step = header[0]
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
    found = []
    for log in _logs_by_age(nc_dir, logs):
        step = None
        for line in log.read_text(errors="replace").splitlines():
            header = _step_header(line)
            if header is not None:
                step = header[0]
            elif any(pat in line for pat in _SOLVER_WARNINGS):
                found.append((log.name, step, line.strip()))
    return found


#: Residual the CGS iteration stops at (``ErroIterSols``,
#: NC_CommonFunctions.cpp:83). A bin that ends *above* this did not converge —
#: which is why it is the reference line on a convergence plot and not a
#: cosmetic annotation.
CGS_TOLERANCE = 1.0e-9

#: Iteration ceiling when ``-nitermax`` is not passed (``niter_max``,
#: NC_CommonFunctions.cpp:84). Reaching it is the first of the two
#: ``_SOLVER_WARNINGS``.
NITER_MAX_DEFAULT = 1500

# "CGS solver: number of iterations = 44, relative error = 4.85659e-10"
# (NC_CommonFunctions.cpp:1222), written once per step whether or not the
# iteration converged.
_CGS_LINE = re.compile(
    r"^CGS solver: number of iterations = (\d+), relative error = (\S+)")

# Lines of the per-step "Time Statistic" block. The same three labels appear
# again in the whole-job block at the end of the log, so they are only read
# while _TIME_HEADER has opened a per-step block — see _parse_log.
_TIME_LABELS = {
    "Assembling the equation system": "t_assemble",
    "Solving the equation system": "t_solve",
    "Post processing": "t_post",
    "Total": "t_total",
}
_TIME_HEADER = "Time Statistic For The Current Step"


@dataclass(frozen=True)
class BinRecord:
    """What one NC*.out log says about one frequency bin.

    Every field but ``bin`` may be None: a log killed mid-step, or fetched
    from a run that died, still names the step it was working on, and a
    half-written block is more useful reported with holes than dropped.
    """

    bin: int                        #: 1-based frequency step
    frequency: float = None         #: Hz, from the step header
    iterations: int = None          #: CGS iterations the bin needed
    rel_error: float = None         #: residual it finished at
    t_assemble: float = None        #: seconds
    t_solve: float = None
    t_post: float = None
    t_total: float = None
    warnings: tuple = ()            #: matching ``_SOLVER_WARNINGS`` lines
    log: str = ""                   #: NC*.out file it was read from

    @property
    def converged(self):
        """True when the bin finished at or below NumCalc's own tolerance.

        None when the log never reported a residual for it — unknown is not
        the same as bad, and a plot must be able to tell them apart.
        """
        if self.warnings:
            return False
        if self.rel_error is None:
            return None
        return self.rel_error <= CGS_TOLERANCE


def _parse_log(log):
    """``{bin: BinRecord}`` for one NC*.out file."""
    out = {}
    step = None
    fields = {}
    timing = False          # inside a per-step Time Statistic block

    def flush():
        if step is not None:
            out[step] = BinRecord(bin=step, log=log.name, **fields)

    for line in log.read_text(errors="replace").splitlines():
        header = _step_header(line)
        if header is not None:
            flush()
            step, freq = header
            fields = {"frequency": freq, "warnings": ()}
            timing = False
            continue
        if step is None:
            continue        # preamble: nothing to attach it to

        if _TIME_HEADER in line:
            timing = True
            continue
        match = _CGS_LINE.match(line)
        if match is not None:
            fields["iterations"] = int(match.group(1))
            try:
                fields["rel_error"] = float(match.group(2))
            except ValueError:  # truncated number in a killed log
                pass
        elif any(pat in line for pat in _SOLVER_WARNINGS):
            # Both warnings live in the step they belong to, one printed
            # before the summary line (max iterations) and one after it
            # (CGS failed), so a step-scoped accumulator catches both.
            fields["warnings"] = fields["warnings"] + (line.strip(),)
        elif timing and ":" in line:
            label, _, value = line.partition(":")
            key = _TIME_LABELS.get(label.strip())
            if key is not None:
                try:
                    fields[key] = float(value)
                except ValueError:
                    pass
                if key == "t_total":
                    timing = False      # per-step block ends at Total
    flush()
    return out


def convergence(nc_dir, logs=None):
    """Per-bin solver diagnostics from NC*.out, as a bin-sorted BinRecord list.

    NumCalc prints one summary line per frequency step — iterations and final
    residual — and nothing per iteration, so this is as fine-grained as a
    convergence history gets without patching the solver.

    Logs are read oldest first and **the newest record that actually reported
    a solve wins**: a resume writes a fresh log over bins an earlier one
    already covered, and the newest is normally the run that produced the
    be.out/ now on disk. Reading them in name order instead would let a stale
    `NC1-45.out` overwrite the `NC34-45.out` that recomputed those bins.

    The "actually reported a solve" half matters as much as the recency. A run
    that is still in flight, or was killed after printing its step header, adds
    a newer but empty record for that bin; letting it win would blank out the
    solve that the results on disk came from. So an iterations-less record
    fills a gap and never displaces a real one.

    Returns [] when the directory holds no logs — routine rather than
    exceptional, since `bem solve --force` deletes them and they are not
    checked in.

    Args:
        nc_dir: variant directory
        logs: restrict to these log file names, in the order given
              (default: every NC*.out, oldest first)
    """
    merged = {}
    for log in _logs_by_age(nc_dir, logs):
        for b, rec in _parse_log(log).items():
            old = merged.get(b)
            if old is None or rec.iterations is not None:
                merged[b] = rec
    return [merged[b] for b in sorted(merged)]
