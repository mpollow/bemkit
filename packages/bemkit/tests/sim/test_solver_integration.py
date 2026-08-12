"""End-to-end check against the real NumCalc binary.

Every other test in the suite either stops at the file we wrote, or drives a
*fake* NumCalc (``tests/test_pipeline.py``). Neither can catch the failure that
matters most for :mod:`pynumcalc.deck`: emitting a deck the solver rejects
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
from pathlib import Path

import numpy as np
import pytest

from bemkit.config import ModelConfig
from pynumcalc import available_bins, load_bins, solver_warnings
from bemkit.sim.export import export_variant

pytestmark = pytest.mark.skipif(
    shutil.which("NumCalc") is None,
    reason="NumCalc not on PATH — real-solver integration check skipped")

#: Resolved from this file, not pytest's rootpath — the suite moved into
#: packages/bemkit/ and rootpath is the monorepo root.
GOLDEN_MODEL = Path(__file__).resolve().parents[1] / "golden" / "model"

#: The golden model is a 4-element tetrahedron with ~1 m edges, so the top of
#: its own 1-4 kHz band is far past what the mesh can integrate: bin 4 dies with
#: "Too many integral points in the theta-direction". That is a property of the
#: fixture mesh, not of the deck, so the check stops below it. If a future
#: golden model changes, this constant moves with it — the fixture fails loudly
#: rather than silently narrowing.
MAX_SOLVABLE_BIN = 3


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Export the golden model, solve its usable bins.

    Returns ``(nc_dir, expected_freqs, stdout)`` where ``expected_freqs`` comes
    from **config.toml**, not from the deck on disk. That distinction is the
    whole point: reading the frequencies back out of the file under test would
    move both sides of the comparison together and pass no matter how badly
    ``Deck.write`` encoded them.
    """
    model = tmp_path_factory.mktemp("solver") / "model"
    shutil.copytree(GOLDEN_MODEL, model)
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


def test_solver_knobs_reach_the_solver(tmp_path):
    """Tier 1's four knobs are positional fields in Main Parameters I/II — a
    value in the wrong column is a different setting entirely, and NumCalc
    would accept it. Assert against NumCalc's own echo of what it parsed.

    cluster_edge_length is 4.0 m because NumCalc rejects a cluster smaller than
    the elements ("Inputed value of cluster diameter for level 0 is too
    small!") and this fixture's elements are ~1 m.
    """
    model = tmp_path / "model"
    shutil.copytree(GOLDEN_MODEL, model)
    cfg = (model / "config.toml").read_text().replace(
        "[bem]",
        "[bem]\ncluster_edge_length = 4.0\npreconditioner = -1\n"
        "min_expansion_terms = 6", 1)
    (model / "config.toml").write_text(cfg)
    export_variant(ModelConfig.from_dir(model), "tetra", verbose=False)

    nc_dir = model / "numcalc" / "tetra"
    proc = subprocess.run(["NumCalc", "-istart", "1", "-iend", "1"],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:]

    log = (nc_dir / "NC1-1.out").read_text()
    assert "Minimum number of terms in the FMM expansion. = 6" in log
    # IPRECOND = -1 selects incomplete-LU with scanning degree 1.
    assert "incomplete LU-decomposition preconditioner" in log


def test_mixed_boundary_condition_types_solve(tmp_path):
    """Tier 2: a driven element plus an absorbing one in the same deck.

    ADMI takes a **normalized** admittance (Y / (rho0*c)), so 0.4 is a real,
    strongly absorbing surface; an SI value here would be wrong by ~410 and
    NumCalc would accept it without complaint.
    """
    from bemkit.geometry.io import read_mesh_triangles

    model = tmp_path / "model"
    shutil.copytree(GOLDEN_MODEL, model)
    verts, tris = read_mesh_triangles(next((model / "mesh").iterdir()))
    anchor = [float(x) for x in verts[tris].mean(axis=1)[2]]
    (model / "config.toml").write_text(
        (model / "config.toml").read_text()
        + "\nabsorber = {{ near = [{:.6f}, {:.6f}, {:.6f}], "
          'type = "admi", value = [0.4, -0.2] }}\n'.format(*anchor))
    export_variant(ModelConfig.from_dir(model), "tetra", verbose=False)

    nc_dir = model / "numcalc" / "tetra"
    text = (nc_dir / "NC.inp").read_text()
    assert "ELEM 0 TO 0 VELO 1.0 -1 0.0 -1" in text
    assert "ELEM 2 TO 2 ADMI 0.4 -1 -0.2 -1" in text

    proc = subprocess.run(["NumCalc", "-istart", "1", "-iend", "1"],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert available_bins(nc_dir) == [1]

    # The absorbing element must not count toward the vibrating area.
    from pynumcalc.readers import read_nc_inp_boundary_conditions
    vib = read_nc_inp_boundary_conditions(nc_dir / "NC.inp")["vibrating"]
    assert list(vib) == [0]


def test_check_normals_catches_the_inconsistently_wound_fixture(solved):
    """Tier 6: -check_normals is wired through and actually bites.

    The golden tetrahedron's triangles are NOT consistently wound — edge 0->1
    appears in both [0,1,2] and [0,1,3] — so NumCalc's half-edge tally fails.
    That makes it a good positive test of the flag: an inconsistent mesh must
    be rejected, not silently solved. (It only checks that all normals point
    into the *same* domain, not that they point outward.)
    """
    nc_dir, _, _ = solved
    proc = subprocess.run(["NumCalc", "-istart", "1", "-iend", "1",
                           "-check_normals"],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode != 0
    assert "differently ordered" in proc.stdout + proc.stderr


def test_solver_flags_are_accepted_by_numcalc(solved):
    """An unrecognised flag makes NumCalc exit(-1), so the ones bemkit emits
    have to be exactly right."""
    from pynumcalc.runner import solver_flags

    nc_dir, _, _ = solved
    opts = solver_flags(nitermax=500, adapt_fmmlength=True)
    assert opts == ("-nitermax", "500", "-adapt_fmmlength")
    proc = subprocess.run(["NumCalc", "-istart", "1", "-iend", "1", *opts],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-1500:]


def test_estimate_ram_reports_a_figure(solved):
    """-estimate_ram runs clustering only and writes Memory.txt
    ("<step> <freq Hz> <GB>", no header)."""
    nc_dir, _, _ = solved
    (nc_dir / "Memory.txt").unlink(missing_ok=True)
    proc = subprocess.run(["NumCalc", "-estimate_ram", "-istart", "1",
                           "-iend", "1"],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-1500:]
    rows = [ln.split() for ln in
            (nc_dir / "Memory.txt").read_text().splitlines() if ln.strip()]
    assert rows and len(rows[0]) >= 3
    assert float(rows[0][2]) > 0


def test_frequency_dependent_admittance_solves(tmp_path):
    """Tier 5: a CURVES-driven ADMI condition.

    The ELEM line carries the constant 1.0 for *both* parts and two curve ids;
    NumCalc multiplies each part by its own curve, so a 0.0 imaginary constant
    would silently kill the imaginary curve.
    """
    from bemkit.geometry.io import read_mesh_triangles

    model = tmp_path / "model"
    shutil.copytree(GOLDEN_MODEL, model)
    verts, tris = read_mesh_triangles(next((model / "mesh").iterdir()))
    anchor = [float(x) for x in verts[tris].mean(axis=1)[2]]
    (model / "config.toml").write_text(
        (model / "config.toml").read_text()
        + "\nabsorber = {{ near = [{:.6f}, {:.6f}, {:.6f}], "
          'type = "admi", curve = "felt" }}\n'.format(*anchor)
        + "\n[curves.felt]\nfrequencies = [100.0, 4000.0]\n"
          "real = [0.1, 0.6]\nimag = [-0.05, -0.3]\n")
    export_variant(ModelConfig.from_dir(model), "tetra", verbose=False)

    nc_dir = model / "numcalc" / "tetra"
    text = (nc_dir / "NC.inp").read_text()
    assert "ELEM 2 TO 2 ADMI 1.0 1 1.0 2" in text
    assert "CURVES\n2 2\n" in text

    proc = subprocess.run(["NumCalc", "-istart", "1", "-iend", "1"],
                          cwd=nc_dir, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert available_bins(nc_dir) == [1]
