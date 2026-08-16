import os
import stat

import pytest

from numcalckit.runner import (
    Job, LocalRunner, missing_ranges, plan_jobs, run_jobs,
)


def _write_bin(be_out, i):
    """Create be.{i}/ with a non-stub pBoundary so it counts as available."""
    d = be_out / f"be.{i}"
    d.mkdir(parents=True)
    (d / "pBoundary").write_text("header1\nheader2\nheader3\n0 1.0 2.0\n")


def _variant(tmp_path, name, n_elements=None):
    """A (name, nc_dir) pair, optionally with an ObjectMesh element count."""
    nc_dir = tmp_path / name
    nc_dir.mkdir(parents=True, exist_ok=True)
    if n_elements is not None:
        obj = nc_dir / "ObjectMesh"
        obj.mkdir(exist_ok=True)
        (obj / "Elements.txt").write_text(f"{n_elements}\n0 1 2 3 0 0 0\n")
    return name, nc_dir


def _bins(job, nf):
    """The 1-based bins a job covers."""
    return list(range(1, nf + 1)) if job.steps is None \
        else list(range(job.steps[0], job.steps[1] + 1))


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
    def test_missing_ranges(self, tmp_path):
        be = tmp_path / "be.out"
        for i in (1, 2, 5):
            _write_bin(be, i)
        assert missing_ranges(tmp_path, 6) == [(3, 4), (6, 6)]


class TestPlanJobs:
    def test_default_one_job_per_variant(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=4)
        assert len(jobs) == 1 and jobs[0].steps is None

    def test_no_split_when_variants_already_fill_the_slots(self, tmp_path):
        pairs = [_variant(tmp_path, n) for n in ("a", "b", "c")]
        jobs = plan_jobs(pairs, nf=8, max_jobs=2)
        # 3 variants, 2 slots: splitting would only add setup overhead.
        assert len(jobs) == 3 and all(j.steps is None for j in jobs)

    def test_splits_one_variant_to_fill_the_slots(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=8, max_jobs=4)
        assert len(jobs) == 4
        covered = sorted(b for j in jobs for b in _bins(j, 8))
        assert covered == list(range(1, 9))       # contiguous, disjoint, complete

    def test_high_frequency_chunks_are_narrower(self, tmp_path):
        # Cost per bin grows with frequency, so equal-cost chunks must not be
        # equal-width: the top of the band carries fewer bins than the bottom.
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=16, max_jobs=4)
        widths = [len(_bins(j, 16)) for j in sorted(jobs, key=lambda j: j.steps)]
        assert widths == sorted(widths, reverse=True) and widths[0] > widths[-1]

    def test_split_capped_by_available_bins(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=2, max_jobs=5)
        assert sorted(j.steps for j in jobs) == [(1, 1), (2, 2)]

    def test_slots_shared_across_variants(self, tmp_path):
        pairs = [_variant(tmp_path, "a"), _variant(tmp_path, "b")]
        jobs = plan_jobs(pairs, nf=8, max_jobs=4)
        assert len(jobs) == 4
        assert sorted(j.name for j in jobs) == ["a", "a", "b", "b"]

    def test_bigger_mesh_gets_more_slots(self, tmp_path):
        pairs = [_variant(tmp_path, "big", n_elements=40000),
                 _variant(tmp_path, "small", n_elements=1000)]
        jobs = plan_jobs(pairs, nf=16, max_jobs=4)
        assert len(jobs) == 4
        assert sum(j.name == "big" for j in jobs) == 3

    def test_missing_only_skips_complete(self, tmp_path):
        name, nc_dir = _variant(tmp_path, "a")
        for i in range(1, 5):
            _write_bin(nc_dir / "be.out", i)
        assert plan_jobs([(name, nc_dir)], nf=4, missing_only=True) == []

    def test_missing_only_never_spans_a_gap(self, tmp_path):
        name, nc_dir = _variant(tmp_path, "a")
        for i in (1, 2, 5):
            _write_bin(nc_dir / "be.out", i)
        jobs = plan_jobs([(name, nc_dir)], nf=6, max_jobs=4, missing_only=True)
        covered = sorted(b for j in jobs for b in _bins(j, 6))
        assert covered == [3, 4, 6]               # bin 5 is done, never re-run

    def test_fan_out_applies_on_a_fresh_resume(self, tmp_path):
        # Nothing computed yet: the resume path must still fill the slots
        # rather than emit one whole-band job.
        name, nc_dir = _variant(tmp_path, "a")
        jobs = plan_jobs([(name, nc_dir)], nf=6, max_jobs=3, missing_only=True)
        assert len(jobs) == 3 and all(j.steps is not None for j in jobs)
        assert sorted(b for j in jobs for b in _bins(j, 6)) == list(range(1, 7))

    def test_fan_out_on_a_partial_resume(self, tmp_path):
        name, nc_dir = _variant(tmp_path, "a")
        for i in (1, 2):
            _write_bin(nc_dir / "be.out", i)
        jobs = plan_jobs([(name, nc_dir)], nf=6, max_jobs=2, missing_only=True)
        assert sorted(j.steps for j in jobs) == [(3, 4), (5, 6)]

    def test_bins_restricts_a_full_run(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=8, bins=[2, 5])
        assert sorted(b for j in jobs for b in _bins(j, 8)) == [2, 5]

    def test_bins_always_get_explicit_steps(self, tmp_path):
        """A partial run must never take the plain-NumCalc shortcut, or it
        would write NC.out and collide with a full run's log."""
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=2, bins=[1, 2])
        assert all(j.steps is not None for j in jobs)

    def test_scattered_bins_stay_scattered(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=8, bins=[1, 4, 8])
        assert sorted(j.steps for j in jobs) == [(1, 1), (4, 4), (8, 8)]

    def test_adjacent_bins_become_one_range(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=8, bins=[3, 4, 5])
        assert [j.steps for j in jobs] == [(3, 5)]

    def test_bins_intersect_with_missing_on_a_resume(self, tmp_path):
        name, nc_dir = _variant(tmp_path, "a")
        for i in (1, 2):
            _write_bin(nc_dir / "be.out", i)
        jobs = plan_jobs([(name, nc_dir)], nf=8, bins=[1, 2, 7],
                         missing_only=True)
        assert sorted(b for j in jobs for b in _bins(j, 8)) == [7]

    def test_bins_already_complete_plans_nothing(self, tmp_path):
        name, nc_dir = _variant(tmp_path, "a")
        for i in (1, 2):
            _write_bin(nc_dir / "be.out", i)
        assert plan_jobs([(name, nc_dir)], nf=8, bins=[1, 2],
                         missing_only=True) == []

    def test_log_names_are_distinct_per_bin(self, tmp_path):
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=8, bins=[1, 4])
        assert sorted(j.log_name for j in jobs) == ["NC1-1.out", "NC4-4.out"]

    def test_scattered_bins_are_not_split_further(self, tmp_path):
        """8 pieces already exceed 4 slots, so -j only throttles them."""
        jobs = plan_jobs([_variant(tmp_path, "a")], nf=16,
                         bins=[1, 3, 5, 7, 9, 11, 13, 15], max_jobs=4)
        assert len(jobs) == 8


class TestJobOrder:
    def test_heavy_and_light_variants_are_interleaved(self, tmp_path):
        # -j 2 runs two jobs side by side; pairing the biggest mesh with the
        # smallest keeps peak memory flatter than running the two biggest.
        pairs = [_variant(tmp_path, n, n_elements=e) for n, e in
                 (("a", 40000), ("b", 30000), ("c", 20000), ("d", 10000))]
        jobs = plan_jobs(pairs, nf=8, max_jobs=2)
        assert [j.name for j in jobs] == ["a", "d", "b", "c"]

    def test_biggest_job_starts_first(self, tmp_path):
        pairs = [_variant(tmp_path, "small", n_elements=1000),
                 _variant(tmp_path, "big", n_elements=40000)]
        jobs = plan_jobs(pairs, nf=8, max_jobs=1)
        assert jobs[0].name == "big"


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


class TestSolverFlags:
    """Extra NumCalc argv (Tier 6). An unrecognised flag is exit(-1), so the
    CLI never forwards a raw string — solver_flags builds the list."""

    def test_empty_by_default(self):
        from numcalckit.runner import solver_flags
        assert solver_flags() == ()

    def test_builds_each_flag(self):
        from numcalckit.runner import solver_flags
        assert solver_flags(nitermax=500) == ("-nitermax", "500")
        assert solver_flags(check_normals=True) == ("-check_normals",)
        assert solver_flags(adapt_fmmlength=True) == ("-adapt_fmmlength",)

    @pytest.mark.parametrize("bad", [0, -5, 1.5, True, "many"])
    def test_rejects_bad_nitermax(self, bad):
        from numcalckit.runner import solver_flags
        with pytest.raises(ValueError, match="positive integer"):
            solver_flags(nitermax=bad)

    def test_options_ride_along_on_every_planned_job(self, tmp_path):
        from numcalckit.runner import plan_jobs
        (tmp_path / "a").mkdir()
        jobs = plan_jobs([("a", tmp_path / "a")], 8,
                         options=("-nitermax", "500"))
        assert jobs and all(j.options == ("-nitermax", "500") for j in jobs)
