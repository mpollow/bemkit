import numpy as np

from pynumcalc.status import (
    available_bins,
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
        import pynumcalc.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [11, 22])
        assert running_variants(tmp_path, ["a"]) == {"a": [11, 22]}

    def test_omits_variants_with_no_live_process(self, tmp_path, monkeypatch):
        import pynumcalc.status as status
        monkeypatch.setattr(status, "numcalc_pids",
                            lambda d: [7] if d.name == "busy" else [])
        assert running_variants(tmp_path, ["idle", "busy"]) == {"busy": [7]}

    def test_looks_under_the_models_numcalc_directory(self, tmp_path, monkeypatch):
        seen = []
        import pynumcalc.status as status
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
        import pynumcalc.status as status
        monkeypatch.setattr(status, "_numcalc_procs", lambda: iter(()))
        assert scan_numcalc() == {}

    def test_groups_pids_by_the_directory_they_run_in(self, monkeypatch, tmp_path):
        import pynumcalc.status as status
        a = tmp_path / "m" / "numcalc" / "v1"
        b = tmp_path / "m" / "numcalc" / "v2"
        monkeypatch.setattr(status, "_numcalc_procs",
                            lambda: iter([(10, a), (12, b), (11, a)]))
        assert scan_numcalc() == {a: [10, 11], b: [12]}

    def test_pids_within_a_directory_are_sorted(self, monkeypatch, tmp_path):
        import pynumcalc.status as status
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
