"""End-to-end check against the real NumCalc binary.

Every other test in the suite either stops at the file we wrote, or drives a
*fake* NumCalc (``tests/test_pipeline.py``). Neither can catch the failure that
matters most for :mod:`bemkit.numcalc.deck`: emitting a deck the solver rejects
or, worse, silently misreads. NumCalc's lexer splits on the space character
only, treats a first term containing ``#`` as a comment, and derives the solved
frequencies from an interpolated curve — all of which are easy to break in a way
no unit test would notice.

Skipped unless NumCalc is on PATH, so it never blocks a normal run. The model is
``tests/golden/model`` (4 nodes, 4 elements), which solves in well under a
second.
"""

import re
import shutil
import subprocess

import numpy as np
import pytest

from bemkit.config import ModelConfig
from bemkit.numcalc import available_bins, load_bins, solver_warnings
from bemkit.sim.export import export_variant

pytestmark = pytest.mark.skipif(
    shutil.which("NumCalc") is None,
    reason="NumCalc not on PATH — real-solver integration check skipped")

GOLDEN_MODEL = "tests/golden/model"

#: The golden model is a 4-element tetrahedron with ~1 m edges, so the top of
#: its own 1-4 kHz band is far past what the mesh can integrate: bin 4 dies with
#: "Too many integral points in the theta-direction". That is a property of the
#: fixture mesh, not of the deck, so the check stops below it. If a future
#: golden model changes, this constant moves with it — the fixture fails loudly
#: rather than silently narrowing.
MAX_SOLVABLE_BIN = 3


@pytest.fixture(scope="module")
def solved(tmp_path_factory, pytestconfig):
    """Export the golden model, solve its usable bins.

    Returns ``(nc_dir, expected_freqs, stdout)`` where ``expected_freqs`` comes
    from **config.toml**, not from the deck on disk. That distinction is the
    whole point: reading the frequencies back out of the file under test would
    move both sides of the comparison together and pass no matter how badly
    ``Deck.write`` encoded them.
    """
    model = tmp_path_factory.mktemp("solver") / "model"
    shutil.copytree(pytestconfig.rootpath / GOLDEN_MODEL, model)
    config = ModelConfig.from_dir(model)
    export_variant(config, "tetra", verbose=False)

    expected_freqs = config.freq_array()
    assert len(expected_freqs) >= MAX_SOLVABLE_BIN

    nc_dir = model / "numcalc" / "tetra"
    proc = subprocess.run(
        ["NumCalc", "-istart", "1", "-iend", str(MAX_SOLVABLE_BIN)],
        cwd=nc_dir, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"NumCalc rejected the deck (exit {proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return nc_dir, expected_freqs, proc.stdout


def test_numcalc_accepts_the_deck(solved):
    """A parse error is an exit(-1) plus an 'Error' line, not a bad result."""
    _, _, stdout = solved
    assert "Error" not in stdout, stdout[-2000:]


def test_numcalc_solves_every_requested_bin(solved):
    nc_dir, _, _ = solved
    assert available_bins(nc_dir) == list(range(1, MAX_SOLVABLE_BIN + 1))


def test_solver_reports_no_convergence_warnings(solved):
    nc_dir, _, _ = solved
    assert solver_warnings(nc_dir) == []


def test_numcalc_solved_the_frequencies_the_deck_asked_for(solved):
    """The load-bearing assertion.

    The deck expresses its frequency list as a Load Frequency Curve that NumCalc
    interpolates back (``NC_ComputeFrequencies``). If that encoding is ever
    broken, the solver happily computes a *different* band and every result is
    silently wrong — no error, no warning, plausible-looking plots.

    Compared against config.toml's grid, so a bug in Deck.write cannot hide by
    also corrupting what Deck.read reports.
    """
    nc_dir, expected_freqs, _ = solved
    logged = {}
    for log in sorted(nc_dir.glob("NC*.out")):
        for step, freq in re.findall(r"Step (\d+), Frequency = ([\d.eE+-]+) Hz",
                                     log.read_text()):
            logged[int(step)] = float(freq)

    assert logged, "no 'Step N, Frequency = ...' lines found in NC*.out"
    assert sorted(logged) == list(range(1, MAX_SOLVABLE_BIN + 1))
    np.testing.assert_allclose(
        [logged[i + 1] for i in range(MAX_SOLVABLE_BIN)],
        expected_freqs[:MAX_SOLVABLE_BIN], rtol=1e-9,
        err_msg="NumCalc solved a different band than config.toml specifies")


def test_results_are_finite_and_driven(solved):
    """A rigid surface with no source solves to silence without complaining, so
    'the file parsed' is not enough — the BC must actually have driven it."""
    nc_dir, expected_freqs, _ = solved
    for quantity in ("pBoundary", "pEvalGrid"):
        # load_bins spans the whole band and leaves unsolved bins NaN, so look
        # only at the bins this fixture actually solved.
        data = load_bins(nc_dir, quantity, len(expected_freqs))
        data = data[:, :MAX_SOLVABLE_BIN]
        assert np.isfinite(data).all(), f"{quantity} contains NaN/Inf"
        assert np.abs(data).max() > 0, (
            f"{quantity} is identically zero — the deck's boundary condition "
            f"did not drive the surface")
