"""NumCalc execution backends.

A variant directory ``numcalc/{variant}/`` is a self-contained, relocatable
job bundle: NumCalc runs with it as CWD (relative paths only) and writes
results into it. Frequency bins are independent (``-istart``/``-iend``) and
``be.out/be.{i}/`` outputs are disjoint, so one variant can fan out over
several processes and merging is a no-op.

``LocalRunner`` is the only implemented backend. Remote backends (SSH,
scheduler, cloud batch) implement the same ``Runner`` interface: transfer
the bundle, start NumCalc, report process state, bring results back. Bin
completion is *not* tracked here — ``bemkit.numcalc.status`` derives it from
``be.out/`` itself.
"""

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from bemkit.numcalc.status import missing_bins


@dataclass(frozen=True)
class Job:
    """One NumCalc invocation: a variant directory and an optional bin range."""

    name: str
    nc_dir: Path
    steps: tuple[int, int] | None = None  # (istart, iend), 1-based inclusive
    #: Extra NumCalc flags, e.g. ("-nitermax", "5000") or ("-check_normals",).
    #: NumCalc exits(-1) on an unrecognised flag, so these are built by
    #: solver_flags() rather than passed through from user input verbatim.
    options: tuple[str, ...] = ()

    @property
    def log_name(self):
        """NumCalc's log file for this job — which is why parallel bin
        ranges in one directory don't clobber each other's logs."""
        if self.steps:
            return f"NC{self.steps[0]}-{self.steps[1]}.out"
        return "NC.out"


class Runner(ABC):
    """Execution backend interface."""

    @abstractmethod
    def submit(self, job: Job):
        """Start a job; returns a backend-specific handle."""

    @abstractmethod
    def poll(self, handle):
        """Return the job's exit code, or None while still running."""

    @abstractmethod
    def fetch(self, job: Job):
        """Make results available in job.nc_dir (no-op for local runs)."""

    @abstractmethod
    def cancel(self, handle):
        """Stop a running job (used to clean up on interrupt/failure)."""


class LocalRunner(Runner):
    """Run NumCalc as local subprocesses (CWD = job.nc_dir)."""

    def submit(self, job):
        cmd = ["NumCalc"]
        if job.steps is not None:
            cmd += ["-istart", str(job.steps[0]), "-iend", str(job.steps[1])]
        cmd += list(job.options)
        return subprocess.Popen(cmd, cwd=str(job.nc_dir),
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)

    def poll(self, handle):
        return handle.poll()

    def fetch(self, job):
        pass

    def cancel(self, handle):
        if handle.poll() is not None:
            return
        handle.terminate()
        try:
            handle.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.kill()
            handle.wait()


#: NumCalc's own flag set (NC_Main.cpp:150-186), minus -istart/-iend which the
#: job planner owns. An unrecognised flag makes NumCalc exit(-1), so the CLI
#: never forwards a raw string.
SOLVER_FLAGS = {
    "nitermax": "-nitermax",            # takes an int
    "check_normals": "-check_normals",  # boolean
    "adapt_fmmlength": "-adapt_fmmlength",  # boolean
}


def solver_flags(*, nitermax=None, check_normals=False, adapt_fmmlength=False):
    """Build the extra NumCalc argv for a run.

    Returns
    -------
    tuple of str
    """
    out = []
    if nitermax is not None:
        if not (isinstance(nitermax, int) and not isinstance(nitermax, bool)
                and nitermax > 0):
            raise ValueError(f"nitermax must be a positive integer, "
                             f"got {nitermax!r}")
        out += [SOLVER_FLAGS["nitermax"], str(nitermax)]
    if check_normals:
        out.append(SOLVER_FLAGS["check_normals"])
    if adapt_fmmlength:
        out.append(SOLVER_FLAGS["adapt_fmmlength"])
    return tuple(out)


def _contiguous(bins):
    """Collapse a sorted bin list into contiguous (istart, iend) ranges."""
    ranges = []
    for b in bins:
        if ranges and b == ranges[-1][1] + 1:
            ranges[-1][1] = b
        else:
            ranges.append([b, b])
    return [tuple(r) for r in ranges]


def missing_ranges(nc_dir, nf):
    """Contiguous (istart, iend) ranges covering bins without results."""
    return _contiguous(missing_bins(nc_dir, nf))


# -- Job planning ---------------------------------------------------------
#
# The caller says how many NumCalc processes may run at once (-j); how the
# work is cut up to fill them is decided here. Both cuts and queue order use
# a crude cost model: a bin costs its own index (NumCalc's per-bin time and
# memory grow with frequency — deeper FMM octree, more multipole terms per
# box), a variant costs its element count. Only ratios matter, and only for
# scheduling — no result ever depends on how the bins were partitioned.


def _element_count(nc_dir):
    """Boundary elements of a variant, or 1 if not exported (yet)."""
    try:
        with open(Path(nc_dir) / "ObjectMesh" / "Elements.txt") as f:
            return max(1, int(f.readline().strip()))
    except (OSError, ValueError):
        return 1


def _cost(rng):
    """Cost of an inclusive bin range: sum of its bin indices."""
    a, b = rng
    return (a + b) * (b - a + 1) / 2


def _bisect(rng):
    """Cut one range in two at its cost midpoint (not its midpoint bin)."""
    a, b = rng
    total, left, best = _cost(rng), 0, None
    for m in range(a, b):
        left += m
        gap = abs(left * 2 - total)
        if best is None or gap < best[0]:
            best = (gap, m)
    return (a, best[1]), (best[1] + 1, b)


def _split_ranges(ranges, n_parts):
    """Cut ranges into n_parts pieces of ~equal cost, never across a gap.

    Fewer pieces come back when the bins run out (a range of one bin is
    indivisible), so a resume with holes in be.out/ keeps its holes.
    """
    parts = list(ranges)
    while len(parts) < n_parts:
        divisible = [p for p in parts if p[1] > p[0]]
        if not divisible:
            break
        worst = max(divisible, key=_cost)
        parts.remove(worst)
        parts.extend(_bisect(worst))
    return sorted(parts)


def _allocate(costs, n_slots, caps):
    """Hand out n_slots among variants by cost, ≥1 and ≤cap each."""
    slots = [1] * len(costs)
    for _ in range(n_slots - len(costs)):
        free = [i for i in range(len(costs)) if slots[i] < caps[i]]
        if not free:
            break
        slots[max(free, key=lambda i: costs[i] / slots[i])] += 1
    return slots


def _queue_order(weighted):
    """Order (cost, job) pairs heaviest-first, alternating heavy and light.

    The heaviest job starts first so it is not left running alone at the end,
    but each pair of adjacent slots mixes a big job with a small one — with
    -j 2, the 40k-element variant runs next to the 8k one and the top-octave
    bin range next to the bottom one, which keeps peak memory flatter than
    two heavyweights side by side.
    """
    ranked = [job for _, job in sorted(weighted, key=lambda p: -p[0])]
    out = []
    lo, hi = 0, len(ranked) - 1
    while lo <= hi:
        out.append(ranked[lo])
        lo += 1
        if lo <= hi:
            out.append(ranked[hi])
            hi -= 1
    return out


def plan_jobs(variants, nf, *, max_jobs=None, missing_only=False, bins=None,
               options=()):
    """Build the job list for a run.

    One job per variant, except that fewer variants than ``max_jobs`` would
    leave processes idle — then variants are fanned out into bin ranges
    (``-istart``/``-iend``) until the slots are filled, the busiest variant
    first. Ranges are cut for equal cost, so the expensive top of the band
    gets narrower slices than the cheap bottom.

    Args:
        variants: list of (name, nc_dir) pairs
        nf: number of frequency bins
        options: extra NumCalc flags applied to every job (see solver_flags)
        max_jobs: NumCalc processes that may run at once (None: no fan-out)
        missing_only: only schedule bins absent from be.out/ (resume)
        bins: 1-based bins to restrict the run to (None: the whole band).
            A partial-band run — intersected with whatever would otherwise
            be scheduled, so it never re-runs a bin that already has results.
    """
    todo = []
    for name, nc_dir in variants:
        nc_dir = Path(nc_dir)
        wanted = (set(missing_bins(nc_dir, nf)) if missing_only
                  else set(range(1, nf + 1)))
        if bins is not None:
            wanted &= set(bins)
        ranges = _contiguous(sorted(wanted))
        if ranges:
            todo.append((name, nc_dir, ranges, _element_count(nc_dir)))
    if not todo:
        return []

    costs = [elem * sum(map(_cost, rng)) for _, _, rng, elem in todo]
    if max_jobs and max_jobs > len(todo):
        caps = [sum(b - a + 1 for a, b in rng) for _, _, rng, _ in todo]
        slots = _allocate(costs, max_jobs, caps)
    else:
        slots = [1] * len(todo)

    weighted = []
    for (name, nc_dir, ranges, elem), n in zip(todo, slots):
        for rng in _split_ranges(ranges, n):
            # A lone full-band range keeps NumCalc's plain invocation (and its
            # NC.out log); a resume or a partial-band run always states its
            # range explicitly, so it can never collide with a full run's log.
            steps = (None if rng == (1, nf) and not missing_only and bins is None
                     else rng)
            weighted.append((elem * _cost(rng),
                             Job(name, nc_dir, steps, options=tuple(options))))
    return _queue_order(weighted)


def run_jobs(jobs, *, runner=None, max_jobs=None, poll_interval=0.5, verbose=True):
    """Run jobs in parallel through a Runner. Returns (completed, failed)."""
    runner = runner or LocalRunner()
    if not jobs:
        return [], []
    max_jobs = max_jobs or len(jobs)
    pending = list(jobs)
    active = {}
    completed, failed = [], []

    try:
        while pending or active:
            while pending and len(active) < max_jobs:
                job = pending.pop(0)
                if verbose:
                    rng = f" bins {job.steps[0]}-{job.steps[1]}" if job.steps else ""
                    print(f"  {job.name}{rng} -> {job.nc_dir / job.log_name}")
                active[runner.submit(job)] = job
            # Poll each handle once per sweep — a remote backend's poll may be
            # a network round-trip, so don't call it twice for the same handle.
            done = [(h, j, rc) for h, j in active.items()
                    if (rc := runner.poll(h)) is not None]
            if not done:
                time.sleep(poll_interval)
                continue
            for handle, job, rc in done:
                del active[handle]
                runner.fetch(job)
                (completed if rc == 0 else failed).append(job)
                if verbose:
                    print(f"  {job.name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'}")
    finally:
        # On KeyboardInterrupt or a submit failure, stop the jobs still running
        # so NumCalc processes are not orphaned to burn CPU in the background.
        for handle in active:
            runner.cancel(handle)
    return completed, failed
