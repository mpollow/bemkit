"""NumCalcSolver.solve end to end -- requires the NumCalc binary on PATH."""

import shutil

import numpy as np
import pytest

from bemkit.model import BC, Group, Medium, Mesh, Model, Velocity, fft_grid, on_axis
from bemkit.results.results import Results
from bemkit.solvers.numcalc import NumCalcSolver

pytestmark = pytest.mark.skipif(
    shutil.which("NumCalc") is None, reason="NumCalc binary not on PATH")


def _sphere_ish():
    """A closed, outward-wound octahedron, 0.2 m across."""
    v = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0],
                  [0, -1, 0], [0, 0, 1.], [0, 0, -1]]) * 0.1
    t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return Mesh(v, t, groups={"vibrating": np.array([0])})


def _model(**kw):
    kw.setdefault("fields", {"onaxis": on_axis("z", [0.5, 1.0])})
    return Model(_sphere_ish(), fft_grid(2000, 3), Medium(),
                 bcs=[BC(Group("vibrating"), Velocity(1.0))], **kw)


def test_bundle_appears_in_workdir(tmp_path):
    NumCalcSolver().solve(_model(), workdir=tmp_path)
    assert (tmp_path / "NC.inp").exists()
    assert (tmp_path / "ObjectMesh" / "Nodes.txt").exists()
    assert (tmp_path / "eval_offsets.json").exists()


def test_pressure_shape_finite_and_nonzero(tmp_path):
    res = NumCalcSolver().solve(_model(), workdir=tmp_path)
    p = res.pressure("onaxis")
    assert p.shape == (2, 2)
    assert np.all(np.isfinite(p))
    assert np.any(p != 0)


def test_frequencies_come_back_unchanged(tmp_path):
    model = _model()
    res = NumCalcSolver().solve(model, workdir=tmp_path)
    np.testing.assert_array_equal(res.frequencies, model.frequencies)


def test_workdir_none_still_returns_finite_materialised_results():
    res = NumCalcSolver().solve(_model(), workdir=None)
    p = res.pressure("onaxis")
    assert p.shape == (2, 2)
    assert np.all(np.isfinite(p))
    # materialised: reading again must not touch a now-deleted temp dir
    assert isinstance(res.store, type(res.store))
    p2 = res.pressure("onaxis")
    np.testing.assert_array_equal(p, p2)


def test_result_is_a_results_instance(tmp_path):
    res = NumCalcSolver().solve(_model(), workdir=tmp_path)
    assert isinstance(res, Results)


def test_all_field_points_dropped_raises_clearly(tmp_path):
    # An off-axis point near the centre of the sphere is inside it -- for an
    # exterior problem (the default) it must be dropped, and with no other
    # points left, solve() must raise instead of handing NumCalc an empty
    # EvalGrid. (Deliberately off-axis: a point exactly on an axis lines up
    # with the octahedron's own vertices and hits the ray-caster's degenerate
    # vertex-graze case.)
    model = _model(fields={"inside": np.array([[0.01, 0.01, 0.01]])})
    with pytest.raises(ValueError, match="dropped"):
        NumCalcSolver().solve(model, workdir=tmp_path)


def test_no_field_points_at_all_raises_clearly(tmp_path):
    model = _model(fields={})
    with pytest.raises(ValueError, match="field point"):
        NumCalcSolver().solve(model, workdir=tmp_path)


def test_status_before_any_export_raises(tmp_path):
    # An empty directory is not a bundle -- there is no NC.inp to read the
    # frequency count from, so "complete" cannot be answered at all (a
    # vacuous "no bins missing" would be actively misleading).
    with pytest.raises(FileNotFoundError):
        NumCalcSolver().status(tmp_path)


def test_partial_band_leaves_nan_not_zero_for_missing_bins(tmp_path):
    """bins= restricts a solve to a subset -- the un-requested bin must come
    back as NaN ('not computed'), never as 0+0j ('computed to silence'),
    since those mean very different things to a caller."""
    model = _model()
    res = NumCalcSolver().solve(model, workdir=tmp_path, bins=[1])
    p = res.pressure("onaxis")
    assert np.all(np.isfinite(p[:, 0]))
    assert np.all(np.isnan(p[:, 1]))


def test_status_after_a_complete_run(tmp_path):
    model = _model()
    NumCalcSolver().solve(model, workdir=tmp_path)
    status = NumCalcSolver().status(tmp_path)
    assert status.complete
    assert status.bins_present == (1, 2)
    assert status.bins_missing == ()
