import numpy as np

from numcalckit.status import (
    CGS_TOLERANCE,
    available_bins,
    convergence,
    current_step,
    is_complete,
    missing_bins,
    numcalc_pids,
    running_variants,
    scan_numcalc,
    solver_warnings,
)


def test_available_bins_from_be_out(numcalc_dir):
    bins = available_bins(numcalc_dir)
    assert bins and bins == sorted(bins)
    assert is_complete(numcalc_dir, nf=len(bins))


def test_missing_bins_complement(numcalc_dir):
    import shutil
    shutil.rmtree(numcalc_dir / "be.out" / "be.1")
    nf = 4
    assert 1 in missing_bins(numcalc_dir, nf)
    assert not is_complete(numcalc_dir, nf)


def test_empty_dir(tmp_path):
    assert available_bins(tmp_path) == []
    assert missing_bins(tmp_path, 3) == [1, 2, 3]


def test_crashed_bin_stub_reported_missing(numcalc_dir):
    # Overwrite bin 2's pBoundary with a <10-byte stub (as NumCalc leaves for a
    # crashed/skipped bin). The directory still exists, but the bin must count
    # as missing so `run --missing-only` re-runs it.
    stub = numcalc_dir / "be.out" / "be.2" / "pBoundary"
    stub.write_text("x\n")
    assert 2 not in available_bins(numcalc_dir)
    assert 2 in missing_bins(numcalc_dir, nf=2)
    assert not is_complete(numcalc_dir, nf=2)


def test_half_written_bin_reported_missing(numcalc_dir):
    # A run killed mid-bin leaves a complete pBoundary next to an empty
    # pEvalGrid (NumCalc writes the boundary files first). The bin must
    # count as missing so `bem run` recomputes it instead of leaving a
    # permanent hole in the eval data.
    (numcalc_dir / "be.out" / "be.2" / "pEvalGrid").write_text("")
    (numcalc_dir / "be.out" / "be.2" / "vEvalGrid").write_text("")
    assert available_bins(numcalc_dir) == [1]
    assert 2 in missing_bins(numcalc_dir, nf=2)


def test_bundle_without_eval_grid_needs_only_boundary(numcalc_dir):
    # No EvalGrid/ in the bundle -> pEvalGrid is not an expected output.
    import shutil
    shutil.rmtree(numcalc_dir / "EvalGrid")
    for bi in (1, 2):
        (numcalc_dir / "be.out" / f"be.{bi}" / "pEvalGrid").unlink()
    assert available_bins(numcalc_dir) == [1, 2]


def test_available_bins_from_npz(tmp_path):
    np.savez(tmp_path / "be.npz", bin_indices=np.array([1, 3], dtype=np.int32))
    assert available_bins(tmp_path) == [1, 3]
    assert missing_bins(tmp_path, 4) == [2, 4]


def test_current_step_reads_newest_log(tmp_path):
    import os
    (tmp_path / "NC.out").write_text("Step 3, Frequency = 750 Hz\n")
    os.utime(tmp_path / "NC.out", (1, 1))  # older
    (tmp_path / "NC5-8.out").write_text(
        "Step 5, Frequency = 1250 Hz\nStep 6, Frequency = 1500 Hz\n")
    assert current_step(tmp_path) == 6


def test_current_step_without_logs(tmp_path):
    assert current_step(tmp_path) is None


def test_numcalc_pids_none_running(tmp_path):
    # No NumCalc has tmp_path as CWD; must be [] (also [] without /proc).
    assert numcalc_pids(tmp_path) == []


class TestRunningVariants:
    """Which of a model's variants have a live NumCalc — the guard `run` and
    `pack` both need, so neither races an in-flight solver."""

    def test_nothing_running_is_an_empty_mapping(self, tmp_path):
        assert running_variants(tmp_path, ["a", "b"]) == {}

    def test_maps_the_variant_name_to_its_pids(self, tmp_path, monkeypatch):
        import numcalckit.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [11, 22])
        assert running_variants(tmp_path, ["a"]) == {"a": [11, 22]}

    def test_omits_variants_with_no_live_process(self, tmp_path, monkeypatch):
        import numcalckit.status as status
        monkeypatch.setattr(status, "numcalc_pids",
                            lambda d: [7] if d.name == "busy" else [])
        assert running_variants(tmp_path, ["idle", "busy"]) == {"busy": [7]}

    def test_looks_under_the_models_numcalc_directory(self, tmp_path, monkeypatch):
        seen = []
        import numcalckit.status as status
        monkeypatch.setattr(status, "numcalc_pids",
                            lambda d: seen.append(d) or [])
        running_variants(tmp_path, ["v1"])
        assert seen == [tmp_path / "numcalc" / "v1"]

    def test_no_variants_needs_no_proc_scan(self, tmp_path):
        assert running_variants(tmp_path, []) == {}


class TestScanNumcalc:
    """Every NumCalc process on the machine, keyed by its variant directory —
    the cross-model view `bem ps` needs, since the CLI is otherwise scoped to
    one model at a time."""

    def test_empty_when_nothing_is_running(self, monkeypatch):
        import numcalckit.status as status
        monkeypatch.setattr(status, "_numcalc_procs", lambda: iter(()))
        assert scan_numcalc() == {}

    def test_groups_pids_by_the_directory_they_run_in(self, monkeypatch, tmp_path):
        import numcalckit.status as status
        a = tmp_path / "m" / "numcalc" / "v1"
        b = tmp_path / "m" / "numcalc" / "v2"
        monkeypatch.setattr(status, "_numcalc_procs",
                            lambda: iter([(10, a), (12, b), (11, a)]))
        assert scan_numcalc() == {a: [10, 11], b: [12]}

    def test_pids_within_a_directory_are_sorted(self, monkeypatch, tmp_path):
        import numcalckit.status as status
        a = tmp_path / "m" / "numcalc" / "v1"
        monkeypatch.setattr(status, "_numcalc_procs",
                            lambda: iter([(30, a), (10, a), (20, a)]))
        assert scan_numcalc()[a] == [10, 20, 30]


_LOG = """\
Reading the input file
Step increment .............................. = 1e-06
Warning: Element with very long side found!

Step 1, Frequency = 250 Hz
all good

Step 2, Frequency = 500 Hz
Warning: Maximum number of iterations is reached!

Step 3, Frequency = 750 Hz
Iteration method CGS failed, direct method is used!
"""


def test_solver_warnings_maps_to_bins(tmp_path):
    (tmp_path / "NC.out").write_text(_LOG)
    issues = solver_warnings(tmp_path)
    assert [(b, "iterations" in line) for _, b, line in issues] == \
        [(2, True), (3, False)]
    assert all(log == "NC.out" for log, _, _ in issues)


def test_solver_warnings_before_first_step(tmp_path):
    (tmp_path / "NC1-2.out").write_text(
        "setup\nIteration method CGS failed, direct method is used!\n")
    assert solver_warnings(tmp_path) == [
        ("NC1-2.out", None, "Iteration method CGS failed, direct method is used!")]


def test_solver_warnings_logs_filter(tmp_path):
    (tmp_path / "NC.out").write_text(_LOG)
    (tmp_path / "NC3-4.out").write_text("Step 4, Frequency = 1000 Hz\nclean\n")
    assert solver_warnings(tmp_path, logs=["NC3-4.out"]) == []
    assert solver_warnings(tmp_path, logs=["NC.out", "missing.out"])


def test_solver_warnings_no_logs(tmp_path):
    assert solver_warnings(tmp_path) == []


def test_solver_warnings_garbled_step_line(tmp_path):
    (tmp_path / "NC.out").write_text(
        "Step ???, Frequency = garbage\n"
        "Warning: Maximum number of iterations is reached!\n")
    assert solver_warnings(tmp_path) == [
        ("NC.out", None, "Warning: Maximum number of iterations is reached!")]


def _step(bin_, freq, iters, err, times=(45, 25, 25, 95), warn_before=None,
          warn_after=None):
    """One NC*.out step block, in NumCalc's own layout."""
    out = [f"\nStep {bin_}, Frequency = {freq} Hz\n",
           "Number of equations = 42\n"]
    if warn_before:
        out.append(warn_before + "\n")
    out.append(f"\nCGS solver: number of iterations = {iters}, "
               f"relative error = {err}\n")
    if warn_after:
        out.append(warn_after + "\n")
    if times:
        a, s, p, t = times
        out += ["\nTime Statistic For The Current Step (In Second)\n\n",
                f"Assembling the equation system         : {a}\n",
                f"Solving the equation system            : {s}\n",
                f"Post processing                        : {p}\n",
                f"Total                                  : {t}\n"]
    return "".join(out)


#: The whole-job block NumCalc appends after the last step. It repeats three
#: of the per-step timing labels, which is exactly the trap this fixture is
#: here to catch.
_WHOLE_JOB = """
>> T I M E   S T A T I S T I C   (In   Second) <<

Input                                  : 0
Assembling the equation system         : 1181
Solving the equation system            : 736
Post processing                        : 1026
The whole job                          : 2943
"""


class TestConvergence:
    """The per-bin numbers NumCalc prints once per step — how hard a bin was
    and what residual it reached. `solver_warnings` only says whether it
    complained; this is the quantity behind that bit."""

    def test_reads_iterations_error_and_timings(self, tmp_path):
        (tmp_path / "NC1-2.out").write_text(
            "preamble\n" + _step(1, 250, 44, "4.85659e-10")
            + _step(2, 500, 42, "4.22913e-10", times=(41, 31, 27, 99)))
        a, b = convergence(tmp_path)
        assert (a.bin, a.frequency, a.iterations) == (1, 250.0, 44)
        assert a.rel_error == 4.85659e-10
        assert (a.t_assemble, a.t_solve, a.t_post, a.t_total) == (45, 25, 25, 95)
        assert (b.bin, b.iterations, b.t_total) == (2, 42, 99)
        assert a.log == "NC1-2.out"

    def test_the_whole_job_block_does_not_overwrite_the_last_step(self, tmp_path):
        """The trailing summary repeats 'Assembling the equation system' etc.
        with job totals; charging them to the final bin would inflate it by
        the whole run."""
        (tmp_path / "NC.out").write_text(_step(1, 250, 44, "1e-10") + _WHOLE_JOB)
        rec, = convergence(tmp_path)
        assert (rec.t_assemble, rec.t_solve, rec.t_post) == (45, 25, 25)

    def test_a_newer_log_wins_for_a_bin_both_cover(self, tmp_path):
        import os
        (tmp_path / "NC1-2.out").write_text(_step(2, 500, 99, "9e-10"))
        os.utime(tmp_path / "NC1-2.out", (1, 1))          # older
        (tmp_path / "NC2-2.out").write_text(_step(2, 500, 42, "4e-10"))
        rec, = convergence(tmp_path)
        assert (rec.iterations, rec.log) == (42, "NC2-2.out")

    def test_a_newer_log_without_a_solve_does_not_blank_an_older_one(self, tmp_path):
        """A run still in flight, or killed after printing its step header,
        adds a newer but empty record. The results on disk came from the older
        one, so the older one is what must be reported."""
        import os
        (tmp_path / "NC1-1.out").write_text(_step(1, 250, 44, "1e-10"))
        os.utime(tmp_path / "NC1-1.out", (1, 1))          # older
        (tmp_path / "NC1-1b.out").write_text("\nStep 1, Frequency = 250 Hz\n")
        rec, = convergence(tmp_path)
        assert (rec.iterations, rec.log) == (44, "NC1-1.out")

    def test_a_step_killed_before_it_solved_is_reported_with_holes(self, tmp_path):
        """Dropping it would hide that the run reached that bin at all."""
        (tmp_path / "NC.out").write_text("\nStep 7, Frequency = 1750 Hz\n")
        rec, = convergence(tmp_path)
        assert (rec.bin, rec.frequency) == (7, 1750.0)
        assert rec.iterations is None and rec.t_total is None
        assert rec.converged is None       # unknown, not "did not converge"

    def test_both_warnings_attach_to_their_own_step(self, tmp_path):
        """One is printed before the summary line and one after it
        (NC_CommonFunctions.cpp:1210, :1228), so a parser that only looks in
        one direction misses half of them."""
        (tmp_path / "NC.out").write_text(
            _step(1, 250, 1500, "3e-04",
                  warn_before="Warning: Maximum number of iterations is reached!")
            + _step(2, 500, 44, "1e-10",
                    warn_after="Iteration method CGS failed, direct method is used!"))
        a, b = convergence(tmp_path)
        assert "Maximum number" in a.warnings[0] and a.converged is False
        assert "CGS failed" in b.warnings[0]
        # b's residual is fine; the warning is what condemns it.
        assert b.rel_error < CGS_TOLERANCE and b.converged is False

    def test_converged_compares_against_numcalcs_own_tolerance(self, tmp_path):
        (tmp_path / "NC.out").write_text(
            _step(1, 250, 44, "9e-10") + _step(2, 500, 44, "9e-08"))
        a, b = convergence(tmp_path)
        assert a.converged is True
        assert b.converged is False

    def test_logs_filter_restricts_what_is_read(self, tmp_path):
        (tmp_path / "NC1-1.out").write_text(_step(1, 250, 44, "1e-10"))
        (tmp_path / "NC2-2.out").write_text(_step(2, 500, 42, "1e-10"))
        assert [r.bin for r in convergence(tmp_path, logs=["NC2-2.out"])] == [2]
        assert convergence(tmp_path, logs=["missing.out"]) == []

    def test_no_logs_is_empty_not_an_error(self, tmp_path):
        assert convergence(tmp_path) == []
