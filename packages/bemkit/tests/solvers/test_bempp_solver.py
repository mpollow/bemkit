"""BemppSolver's own contract: what it solves, and what it refuses to.

The comparison against NumCalc and the closed form lives in
``test_cross_validation.py``. What is pinned here is everything that must
hold *before* the numbers are worth comparing: that the backend refuses, by
name, every part of a Model it cannot represent; that it never fabricates a
run status for a run it did not persist; and that it shapes its Results the
way ``NumCalcSolver`` does so the two are interchangeable downstream.
"""

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("bempp_cl.api")

from bemkit.model import (  # noqa: E402
    BC, Admittance, Elements, Group, Medium, Mesh, Model, PlaneWave,
    PointSource, Pressure, Velocity, fft_grid, on_axis,
)
from bemkit.results.results import Results  # noqa: E402
from bemkit.solvers.bempp import BemppSolver  # noqa: E402


def _octahedron():
    """A closed, outward-wound octahedron, 0.2 m across."""
    v = np.array([[1., 0, 0], [-1, 0, 0], [0, 1., 0],
                  [0, -1, 0], [0, 0, 1.], [0, 0, -1]]) * 0.1
    t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return Mesh(v, t, groups={"vibrating": np.array([0])})


def _model(**kw):
    kw.setdefault("fields", {"onaxis": on_axis("z", [0.5, 1.0])})
    kw.setdefault("bcs", [BC(Group("vibrating"), Velocity(1.0))])
    return Model(_octahedron(), fft_grid(2000, 3), Medium(), **kw)


# --- the module must stay importable without bempp installed -------------

def test_importing_the_module_does_not_import_bempp():
    """``import bemkit`` reaches this module eagerly, so a top-level
    ``import bempp_cl`` here would make bempp a hard dependency of the whole
    package — and a multi-second one at that."""
    code = ("import sys, bemkit.solvers.bempp; "
            "sys.exit('bempp_cl' if 'bempp_cl' in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"importing bemkit.solvers.bempp pulled in bempp_cl: "
        f"{r.stderr.decode() or r.stdout.decode()}")


# --- what it refuses ------------------------------------------------------

def test_an_interior_problem_is_refused_by_name():
    model = _model(exterior=False)
    with pytest.raises(NotImplementedError, match="exterior"):
        BemppSolver().solve(model)


@pytest.mark.parametrize("condition, name", [
    (Pressure(0.0), "Pressure"),
    (Admittance(1.0), "Admittance"),
])
def test_non_velocity_conditions_are_refused_by_name(condition, name):
    model = _model(bcs=[BC(Group("vibrating"), condition)])
    with pytest.raises(NotImplementedError, match=name):
        BemppSolver().solve(model)


def test_a_frequency_dependent_condition_is_refused():
    """Of the shipped conditions only Admittance/Impedance may vary with
    frequency, and both are already refused for their *kind* — so the
    frequency guard is reached through a custom Condition subclass, which is
    the only way a VELO-kind spectrum can exist at all."""
    class _SweptVelocity(Velocity):
        frequency_dependent_allowed = True

    model = _model(bcs=[BC(Group("vibrating"),
                           _SweptVelocity(lambda f: np.full(len(f), 0.5)))])
    with pytest.raises(NotImplementedError, match="frequency-dependent"):
        BemppSolver().solve(model)


@pytest.mark.parametrize("source", [
    PlaneWave(direction=(0, 0, 1)),
    PointSource(position=(0, 0, 1.0)),
])
def test_incident_sources_are_refused(source):
    model = _model(sources=[source])
    with pytest.raises(NotImplementedError, match="incident sources"):
        BemppSolver().solve(model)


def test_an_inward_wound_mesh_is_refused_rather_than_solved_inside_out():
    """bempp takes normals from the vertex winding, so an inverted mesh
    assembles happily and returns the interior solution under an exterior
    label — the one failure mode that looks like a plausible answer."""
    mesh = _octahedron()
    flipped = Mesh(mesh.vertices, mesh.triangles[:, ::-1],
                   groups={"vibrating": np.array([0])})
    model = Model(flipped, fft_grid(2000, 3), Medium(),
                  bcs=[BC(Group("vibrating"), Velocity(1.0))],
                  fields={"onaxis": on_axis("z", [0.5])})
    with pytest.raises(ValueError, match="outward-wound"):
        BemppSolver().solve(model)


def test_a_boundary_condition_that_selects_nothing_raises():
    model = _model(bcs=[BC(Elements([]), Velocity(1.0))])
    with pytest.raises(ValueError, match="selects no elements"):
        BemppSolver().solve(model)


def test_two_velocity_conditions_on_one_element_raise():
    model = _model(bcs=[BC(Elements([0, 1]), Velocity(1.0)),
                        BC(Elements([1]), Velocity(2.0))])
    with pytest.raises(ValueError, match="two VELO"):
        BemppSolver().solve(model)


def test_bins_outside_the_band_raise():
    with pytest.raises(ValueError, match="outside 1..2"):
        BemppSolver().solve(_model(), bins=[1, 7])


def test_all_field_points_dropped_raises_clearly():
    model = _model(fields={"inside": np.array([[0.01, 0.01, 0.01]])})
    with pytest.raises(ValueError, match="dropped"):
        BemppSolver().solve(model)


# --- status: no workdir, so no status ------------------------------------

def test_status_refuses_rather_than_inventing_a_complete_run(tmp_path):
    """``Status()`` with no bins would report ``complete`` — a run that never
    happened, described as finished. Refusing is the honest answer."""
    with pytest.raises(NotImplementedError, match="no working directory"):
        BemppSolver().status(tmp_path)


def test_workdir_is_ignored_and_nothing_is_written(tmp_path):
    BemppSolver().solve(_model(), workdir=tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- shape parity with NumCalcSolver --------------------------------------

def test_results_have_the_numcalc_shapes_and_quantities():
    model = _model()
    res = BemppSolver().solve(model)

    assert isinstance(res, Results)
    assert res.set_names == ["onaxis"]
    assert sorted(res.store.available()) == [
        "pBoundary", "pEvalGrid", "vBoundary", "vEvalGrid"]
    assert res.pressure("onaxis").shape == (2, 2)
    assert res.velocity("onaxis").shape == (2, 2, 3)
    assert res.boundary_pressure().shape == (model.mesh.n_elements, 2)
    assert res.boundary_velocity().shape == (model.mesh.n_elements, 2)
    np.testing.assert_array_equal(res.frequencies, model.frequencies)


def test_boundary_velocity_echoes_the_prescribed_condition():
    res = BemppSolver().solve(_model())
    v = res.boundary_velocity()[:, 0]
    assert v[0] == 1.0            # the one vibrating element
    assert np.all(v[1:] == 0.0)   # everything else is sound-hard


def test_a_partial_solve_leaves_the_other_bins_nan_not_zero():
    """Same contract as NumCalcSolver: an uncomputed bin is NaN, because
    0+0j means 'solved, and the answer is silence'."""
    res = BemppSolver().solve(_model(), bins=[1])
    p = res.pressure("onaxis")
    assert np.all(np.isfinite(p[:, 0]))
    assert np.all(np.isnan(p[:, 1]))
    assert np.all(np.isnan(res.boundary_pressure()[:, 1]))


def test_a_model_with_no_field_points_still_solves_the_boundary():
    """NumCalc needs a non-empty EvalGrid; an in-process solve does not, and
    claiming otherwise would be borrowing NumCalc's limitation."""
    res = BemppSolver().solve(_model(fields={}))
    assert res.set_names == []
    assert res.pressure().shape == (0, 2)
    assert np.all(np.isfinite(res.boundary_pressure()))


def test_results_survive_a_save_load_round_trip(tmp_path):
    res = BemppSolver().solve(_model())
    before = res.pressure("onaxis")
    after = Results.load(res.save(tmp_path / "results.npz")).pressure("onaxis")
    np.testing.assert_array_equal(before, after)
