import os
import stat

import pytest

from bemkit.sim.runner import (
    Job, LocalRunner, missing_ranges, plan_jobs, run_jobs, split_steps,
)


def _write_bin(be_out, i):
    """Create be.{i}/ with a non-stub pBoundary so it counts as available."""
    d = be_out / f"be.{i}"
    d.mkdir(parents=True)
    (d / "pBoundary").write_text("header1\nheader2\nheader3\n0 1.0 2.0\n")


@pytest.fixture()
def fake_numcalc(tmp_path, monkeypatch):
    """Put a fake NumCalc on PATH that records its args into args.txt (CWD)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "NumCalc"
    exe.write_text('#!/bin/sh\necho "$@" > args.txt\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return exe


class TestJob:
    def test_log_name_full_run(self, tmp_path):
        assert Job("v", tmp_path).log_name == "NC.out"

    def test_log_name_ranged(self, tmp_path):
        assert Job("v", tmp_path, steps=(3, 7)).log_name == "NC3-7.out"


class TestSplitting:
    def test_split_steps_covers_all_bins(self):
        ranges = split_steps(10, 3)
        assert ranges[0][0] == 1 and ranges[-1][1] == 10
        covered = [b for s, e in ranges for b in range(s, e + 1)]
        assert covered == list(range(1, 11))

    def test_split_more_parts_than_bins(self):
        assert split_steps(2, 5) == [(1, 1), (2, 2)]

    def test_missing_ranges(self, tmp_path):
        be = tmp_path / "be.out"
        for i in (1, 2, 5):
            _write_bin(be, i)
        assert missing_ranges(tmp_path, 6) == [(3, 4), (6, 6)]


class TestPlanJobs:
    def test_default_one_job_per_variant(self, tmp_path):
        jobs = plan_jobs([("a", tmp_path)], nf=4)
        assert len(jobs) == 1 and jobs[0].steps is None

    def test_split(self, tmp_path):
        jobs = plan_jobs([("a", tmp_path)], nf=4, split=2)
        assert [j.steps for j in jobs] == [(1, 2), (3, 4)]

    def test_missing_only_skips_complete(self, tmp_path):
        for i in range(1, 5):
            _write_bin(tmp_path / "be.out", i)
        assert plan_jobs([("a", tmp_path)], nf=4, missing_only=True) == []


class TestRunJobs:
    def test_runs_and_passes_step_flags(self, tmp_path, fake_numcalc):
        nc = tmp_path / "v1"
        nc.mkdir()
        completed, failed = run_jobs(
            [Job("v1", nc, steps=(2, 3))], runner=LocalRunner(), verbose=False)
        assert [j.name for j in completed] == ["v1"] and failed == []
        assert (nc / "args.txt").read_text().split() == ["-istart", "2", "-iend", "3"]

    def test_failure_collected(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        exe = bindir / "NumCalc"
        exe.write_text("#!/bin/sh\nexit 3\n")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        nc = tmp_path / "v1"
        nc.mkdir()
        completed, failed = run_jobs([Job("v1", nc)], verbose=False)
        assert completed == [] and [j.name for j in failed] == ["v1"]

    def test_interrupt_cancels_active_jobs(self, tmp_path):
        # A runner whose poll raises mid-loop simulates KeyboardInterrupt.
        # run_jobs must cancel every still-active handle in its finally block.
        cancelled = []

        class BoomRunner:
            def submit(self, job):
                return object()          # opaque handle, never completes

            def poll(self, handle):
                raise KeyboardInterrupt

            def fetch(self, job):
                pass

            def cancel(self, handle):
                cancelled.append(handle)

        nc = tmp_path / "v1"
        nc.mkdir()
        with pytest.raises(KeyboardInterrupt):
            run_jobs([Job("v1", nc)], runner=BoomRunner(), verbose=False)
        assert len(cancelled) == 1  # the submitted job was cancelled

    def test_local_runner_cancel_terminates_process(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        exe = bindir / "NumCalc"
        exe.write_text("#!/bin/sh\nsleep 30\n")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        nc = tmp_path / "v1"
        nc.mkdir()
        runner = LocalRunner()
        handle = runner.submit(Job("v1", nc))
        assert runner.poll(handle) is None   # still running
        runner.cancel(handle)
        assert handle.poll() is not None      # terminated
