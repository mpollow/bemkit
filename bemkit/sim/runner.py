"""NumCalc execution backends.

A variant directory ``numcalc/{variant}/`` is a self-contained, relocatable
job bundle: NumCalc runs with it as CWD (relative paths only) and writes
results into it. Frequency bins are independent (``-istart``/``-iend``) and
``be.out/be.{i}/`` outputs are disjoint, so one variant can fan out over
several processes and merging is a no-op.

``LocalRunner`` is the only implemented backend. Remote backends (SSH,
scheduler, cloud batch) implement the same ``Runner`` interface: transfer
the bundle, start NumCalc, report process state, bring results back. Bin
completion is *not* tracked here — ``bemkit.sim.status`` derives it from
``be.out/`` itself.
"""

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from bemkit.sim.status import missing_bins


@dataclass(frozen=True)
class Job:
    """One NumCalc invocation: a variant directory and an optional bin range."""

    name: str
    nc_dir: Path
    steps: tuple[int, int] | None = None  # (istart, iend), 1-based inclusive

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


def _contiguous(bins):
    """Collapse a sorted bin list into contiguous (istart, iend) ranges."""
    ranges = []
    for b in bins:
        if ranges and b == ranges[-1][1] + 1:
            ranges[-1][1] = b
        else:
            ranges.append([b, b])
    return [tuple(r) for r in ranges]


def split_bins(bins, n_parts):
    """Split a bin list into n_parts near-equal groups of (istart, iend).

    Groups are split at gaps in ``bins``, so a job never spans a bin that
    already has results — a resume fans out without recomputing anything.
    That means len(result) >= n_parts when the missing bins aren't contiguous.
    """
    bins = list(bins)
    if not bins:
        return []
    n_parts = max(1, min(n_parts, len(bins)))
    bounds = [round(i * len(bins) / n_parts) for i in range(n_parts + 1)]
    ranges = []
    for i in range(n_parts):
        ranges += _contiguous(bins[bounds[i]:bounds[i + 1]])
    return ranges


def split_steps(nf, n_parts):
    """Split bins 1..nf into n_parts contiguous (istart, iend) ranges."""
    if nf <= 0:
        return []
    return split_bins(range(1, nf + 1), n_parts)


def missing_ranges(nc_dir, nf):
    """Contiguous (istart, iend) ranges covering bins without results."""
    return _contiguous(missing_bins(nc_dir, nf))


def plan_jobs(variants, nf, *, split=None, missing_only=False):
    """Build the job list for a run.

    Args:
        variants: list of (name, nc_dir) pairs
        nf: number of frequency bins
        split: fan each variant out into this many bin-range jobs (applies to
            the missing bins when missing_only, so resumes fan out too)
        missing_only: only schedule bins absent from be.out/ (resume)
    """
    jobs = []
    for name, nc_dir in variants:
        nc_dir = Path(nc_dir)
        if missing_only:
            steps = split_bins(missing_bins(nc_dir, nf), split or 1)
            jobs += [Job(name, nc_dir, s) for s in steps]
        elif split and split > 1:
            jobs += [Job(name, nc_dir, steps) for steps in split_steps(nf, split)]
        else:
            jobs.append(Job(name, nc_dir))
    return jobs


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
