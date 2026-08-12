"""Port sphcap onto the new spine and prove it end to end.

sphcap is the reference case (rigid sphere, vibrating cap) declared as
Python -- models/sphcap/model.py + params.toml -- instead of a config.toml.
This exercises the whole chain: load_study -> Model -> NumCalcSolver ->
Results, on a real mesh with a real BOUNDARY condition.

Requires the NumCalc binary on PATH. Only ever solves a single bin: the full
64-frequency band across three meshes would take minutes, and that isn't
what this test is checking.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from bemkit.cli.study import load_study
from bemkit.results.results import Results
from bemkit.solvers.numcalc import NumCalcSolver

pytestmark = pytest.mark.skipif(
    shutil.which("NumCalc") is None, reason="NumCalc binary not on PATH")

# packages/bemkit/tests/test_end_to_end.py -> tests -> bemkit -> packages -> repo root
STUDY = Path(__file__).resolve().parents[3] / "models" / "sphcap"


def test_study_has_the_three_mesh_variants():
    variants = load_study(STUDY)
    assert sorted(variants) == ["coarse", "fine", "standard"]


def test_solving_coarse_variant_gives_finite_nonzero_onaxis_pressure(tmp_path):
    variants = load_study(STUDY)
    model = variants["coarse"]

    res = NumCalcSolver().solve(model, workdir=tmp_path, bins=[1])
    p = res.pressure("onaxis")

    assert p.shape == (6, 64)
    assert np.all(np.isfinite(p[:, 0]))
    assert np.any(p[:, 0] != 0)
    # unsolved bins stay NaN, not silently zero
    assert np.all(np.isnan(p[:, 1]))


def test_results_round_trip_through_save_and_load(tmp_path):
    variants = load_study(STUDY)
    model = variants["coarse"]

    res = NumCalcSolver().solve(model, workdir=tmp_path, bins=[1])
    p_before = res.pressure("onaxis")

    saved_path = res.save(tmp_path / "results.npz")
    loaded = Results.load(saved_path)
    p_after = loaded.pressure("onaxis")

    np.testing.assert_array_equal(p_before, p_after)
    assert np.all(np.isfinite(p_after[:, 0]))
    assert np.any(p_after[:, 0] != 0)
