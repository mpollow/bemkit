import numpy as np

from pynumcalc.sim.status import (
    available_bins,
    is_complete,
    missing_bins,
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


def test_available_bins_from_npz(tmp_path):
    np.savez(tmp_path / "be.npz", bin_indices=np.array([1, 3], dtype=np.int32))
    assert available_bins(tmp_path) == [1, 3]
    assert missing_bins(tmp_path, 4) == [2, 4]


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
