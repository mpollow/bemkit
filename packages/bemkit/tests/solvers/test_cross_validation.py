"""Two independent solvers and a closed form, on one neutral Model.

``Model`` is only *called* solver-neutral until something other than NumCalc
has read one. This module is that something: the same Model object goes to
``NumCalcSolver`` (an external binary driven through NC.inp) and to
``BemppSolver`` (dense Galerkin matrices assembled in this process), and both
are checked against ``bemkit.model.analytic.sphcap_pressure``, which is a
spherical-harmonic series that knows about neither.

Three-way agreement is what makes it evidence. Two solvers agreeing could be
two translations of the same misreading of the Model; a solver agreeing with
the closed form could be luck at one point. All three agreeing, on 78 field
points at two frequencies, is not.

**The tolerances are measured, not guessed.** On the coarse sphcap mesh at
250 and 500 Hz the residuals are:

    NumCalc vs closed form   5.3 %
    bempp   vs closed form   5.8 %
    NumCalc vs bempp         1.6 %

The solver-vs-analytic figures are dominated by a single, identified,
*shared* cause and are not solver error: the mesh's vibrating group is 25
flat triangles whose total area is 5.6 % less than the spherical cap they
approximate, and at ka = 0.46 the radiated field is proportional to the
driven area. Both solvers see the same flat triangles, which is why they
agree with each other twenty times more closely than either agrees with the
sphere. ``test_the_analytic_tolerance_is_not_vacuous`` pins that the 7.5 %
band is still narrow enough to resolve the cap angle to about a degree.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bempp_cl.api")
pytest.importorskip("scipy.special")

from bemkit.model import Model  # noqa: E402
from bemkit.model.analytic import sphcap_pressure  # noqa: E402
from bemkit.solvers.bempp import BemppSolver  # noqa: E402
from bemkit.solvers.numcalc import NumCalcSolver  # noqa: E402
from bemkit.study import load_study  # noqa: E402

#: tests/solvers -> tests -> bemkit -> packages -> repo root
STUDY = Path(__file__).resolve().parents[4] / "models" / "sphcap"

#: The coarse variant (564 elements) at bins 1 and 2 — 250 and 500 Hz, i.e.
#: ka = 0.46 and 0.92 on a 50 mm sphere. Low enough that a 564-element mesh
#: resolves the field (~40 elements per wavelength), and a factor of seven
#: below the direct formulation's first irregular frequency (c/2a =
#: 3400 Hz), which is what lets BemppSolver skip Burton-Miller.
VARIANT = "coarse"
BINS = [1, 2]
#: The Model keeps four frequencies but only two are solved, so "the rest is
#: NaN" is something the parity test can actually observe.
N_FREQUENCIES = 4
#: 0-based columns of the solved bins.
COMPUTED = np.array(BINS) - 1

#: Geometry of the closed form, checked against the mesh itself below.
CAP_ANGLE_DEG = 20.0
RADIUS_MM = 50.0
NMAX = 60

#: Measured 0.053 / 0.058; see the module docstring for why it is this wide.
ANALYTIC_RTOL = 0.075
#: Measured 0.0164. The strong one: two independent discretizations of the
#: same Model, with no shared code below `bemkit.model`.
SOLVER_RTOL = 0.025
#: Measured 0.0217. Velocity is a differenced potential in bempp and a
#: dedicated NumCalc output, so it is looser than pressure by construction.
VELOCITY_RTOL = 0.035

needs_numcalc = pytest.mark.skipif(
    shutil.which("NumCalc") is None, reason="NumCalc binary not on PATH")

#: Run a test once per backend, naming the fixture rather than the object so
#: the NumCalc half can carry its own skip.
EITHER_SOLVER = pytest.mark.parametrize("backend", [
    "bempp", pytest.param("numcalc", marks=needs_numcalc)])


def _relative(got, want):
    """Max pointwise |got - want| / |want|, over the solved bins only.

    Both (N, nf) pressure and (N, nf, 3) velocity index the same way on
    axis 1, and the unsolved bins are NaN in the solvers by design.
    """
    got, want = got[:, COMPUTED], want[:, COMPUTED]
    return float(np.max(np.abs(got - want) / np.abs(want)))


@pytest.fixture(scope="module")
def model():
    """The sphcap coarse variant, cut down to the two bins under test.

    A fresh Model rather than ``m.frequencies = m.frequencies[:2]`` on the
    loaded one: the attribute is writable and nothing caches the length, so
    the mutation would work — but it skips ``Model.__init__``'s ascending and
    non-empty checks, and it would silently reshape a fixture other tests in
    the session share. Building one costs nothing.
    """
    loaded = load_study(STUDY)[VARIANT]
    return Model(loaded.mesh,
                 loaded.frequencies[:N_FREQUENCIES],
                 loaded.medium,
                 bcs=list(loaded.bcs),
                 fields=dict(loaded.fields),
                 exterior=loaded.exterior,
                 time_factor=loaded.time_factor)


@pytest.fixture(scope="module")
def analytic(model):
    """``{set name: (N, nf)}`` exact pressure for the same field points."""
    return {name: sphcap_pressure(points, model.frequencies,
                                  c0=model.medium.c, rho0=model.medium.rho,
                                  angle_deg=CAP_ANGLE_DEG,
                                  radius_mm=RADIUS_MM, nmax=NMAX)
            for name, points in model.fields.items()}


@pytest.fixture(scope="module")
def bempp(model):
    return BemppSolver().solve(model, bins=BINS)


@pytest.fixture(scope="module")
def numcalc(model, tmp_path_factory):
    if shutil.which("NumCalc") is None:
        pytest.skip("NumCalc binary not on PATH")
    workdir = tmp_path_factory.mktemp("numcalc")
    return NumCalcSolver().solve(model, workdir=workdir, bins=BINS)


# --- the closed form has to describe *this* mesh --------------------------

def test_the_mesh_is_the_sphere_the_closed_form_assumes(model):
    """``radius_mm=50`` is a parameter name, not a fact about the mesh."""
    radii = np.linalg.norm(model.mesh.vertices, axis=1)
    np.testing.assert_allclose(radii, RADIUS_MM / 1e3, rtol=1e-9)


def test_the_driven_group_is_the_cap_the_closed_form_assumes(model):
    """The one way all three comparisons could pass while measuring the wrong
    physics: if the mesh's vibrating group did not subtend ``CAP_ANGLE_DEG``,
    both solvers would still agree with each other and both would be checked
    against a cap that is not there.

    Measured from the mesh: the driven group's vertices must reach the cap
    edge and stop, and the rigid remainder must start there.
    """
    vertices = model.mesh.vertices
    colatitude = np.degrees(np.arccos(
        np.clip(vertices[:, 2] / np.linalg.norm(vertices, axis=1), -1, 1)))

    driven = np.unique(model.mesh.triangles[model.mesh.groups["vibrating"]])
    rigid = np.unique(model.mesh.triangles[model.mesh.groups["rigid"]])

    assert colatitude[driven].max() == pytest.approx(CAP_ANGLE_DEG, abs=1e-6)
    assert colatitude[rigid].min() == pytest.approx(CAP_ANGLE_DEG, abs=1e-6)


def test_the_faceted_cap_explains_the_residual(model):
    """Why the solver-vs-analytic band is 7.5 % and not 1 %.

    The 25 flat triangles of the driven group have less area than the
    spherical cap they approximate, and at ka < 1 the radiated field scales
    with driven area. This pins that deficit at the ~5 % that both solvers
    then show — so if the mesh is ever refined and the residual stays at 5 %,
    this test stops agreeing that the mesh is the reason.
    """
    radius = RADIUS_MM / 1e3
    faceted = model.mesh.areas[model.mesh.groups["vibrating"]].sum()
    spherical = (2 * np.pi * radius ** 2
                 * (1 - np.cos(np.radians(CAP_ANGLE_DEG))))
    deficit = 1 - faceted / spherical
    assert 0.04 < deficit < 0.07, f"cap area deficit is now {deficit:.1%}"


@needs_numcalc
def test_the_residual_is_the_cap_area_deficit_and_nothing_else(tmp_path):
    """The previous test asserts the explanation; this one measures it.

    sphcap ships three meshes of the same sphere. Refining it shrinks the
    driven group's area deficit, and if that deficit really is the whole
    residual then the residual must shrink with it — in step, not merely in
    the same direction. Solved at 250 Hz with the six on-axis points, the
    ratio residual/deficit comes out at 0.94 (coarse) and 0.96 (standard);
    the fine mesh continues it at 0.37 % deficit and 0.32 % residual, and is
    left out here only because it costs 18 s to solve.

    Were any part of the residual actual solver error, the ratio would climb
    as the deficit fell away underneath it.
    """
    radius = RADIUS_MM / 1e3
    spherical = (2 * np.pi * radius ** 2
                 * (1 - np.cos(np.radians(CAP_ANGLE_DEG))))

    for variant in ("coarse", "standard"):
        loaded = load_study(STUDY)[variant]
        one_bin = Model(loaded.mesh, loaded.frequencies[:1], loaded.medium,
                        bcs=list(loaded.bcs),
                        fields={"onaxis": loaded.fields["onaxis"]})
        got = NumCalcSolver().solve(one_bin, workdir=tmp_path / variant,
                                    bins=[1]).pressure("onaxis")
        want = sphcap_pressure(one_bin.fields["onaxis"], one_bin.frequencies,
                               c0=one_bin.medium.c, rho0=one_bin.medium.rho,
                               angle_deg=CAP_ANGLE_DEG, radius_mm=RADIUS_MM,
                               nmax=NMAX)

        residual = float(np.max(np.abs(got - want) / np.abs(want)))
        deficit = 1 - (loaded.mesh.areas[loaded.mesh.groups["vibrating"]].sum()
                       / spherical)
        assert 0.7 < residual / deficit < 1.3, (
            f"{variant}: residual {residual:.2%} against a cap-area deficit "
            f"of {deficit:.2%} — the deficit no longer accounts for it")


# --- the three comparisons ------------------------------------------------

@needs_numcalc
def test_numcalc_matches_the_closed_form(model, numcalc, analytic):
    for name in sorted(model.fields):
        error = _relative(numcalc.pressure(name), analytic[name])
        assert error < ANALYTIC_RTOL, f"{name}: {error:.1%}"


def test_bempp_matches_the_closed_form(model, bempp, analytic):
    for name in sorted(model.fields):
        error = _relative(bempp.pressure(name), analytic[name])
        assert error < ANALYTIC_RTOL, f"{name}: {error:.1%}"


@needs_numcalc
def test_the_two_solvers_match_each_other(model, numcalc, bempp):
    """The load-bearing test: one Model, two backends sharing no code below
    ``bemkit.model``, agreeing on 78 field points at two frequencies."""
    for name in sorted(model.fields):
        error = _relative(bempp.pressure(name), numcalc.pressure(name))
        assert error < SOLVER_RTOL, f"{name}: {error:.1%}"


@needs_numcalc
def test_the_two_solvers_agree_on_particle_velocity_too(model, numcalc, bempp):
    """Compared as vectors: on the z axis the x and y components are ~1e-8
    against a ~1e-2 z component, so a per-component relative error there is
    noise divided by noise.
    """
    for name in sorted(model.fields):
        want = numcalc.velocity(name)[:, COMPUTED]
        got = bempp.velocity(name)[:, COMPUTED]
        error = float(np.max(np.linalg.norm(got - want, axis=-1)
                             / np.linalg.norm(want, axis=-1)))
        assert error < VELOCITY_RTOL, f"{name}: {error:.1%}"


# --- the tolerances have to be able to fail -------------------------------

@EITHER_SOLVER
@pytest.mark.parametrize("wrong_angle", [18.0, 21.0])
def test_the_analytic_tolerance_is_not_vacuous(request, model, backend,
                                               wrong_angle):
    """A 7.5 % band is wide. This is what it still catches.

    Compared against the closed form for a cap one to two degrees off, both
    solvers land 14-17 % out — so the band that passes above resolves the
    driven area to roughly a degree of cap angle. A backend that solved a
    plausible-but-different problem would have to land inside that.
    """
    solver = request.getfixturevalue(backend)
    wrong = sphcap_pressure(model.fields["onaxis"], model.frequencies,
                            c0=model.medium.c, rho0=model.medium.rho,
                            angle_deg=wrong_angle, radius_mm=RADIUS_MM,
                            nmax=NMAX)
    assert _relative(solver.pressure("onaxis"), wrong) > ANALYTIC_RTOL


@EITHER_SOLVER
def test_no_solver_output_needs_conjugating(request, backend, analytic):
    """Both solvers are already on the closed form's time convention.

    bempp's kernel is ``exp(+ikr)/(4*pi*r)`` and bemkit's convention is the
    other one, so ``BemppSolver`` conjugates on the way out. That is a
    deliberate, documented conversion — and this is what stops it from being
    a fudge factor someone flips to make a test pass: conjugating a solver's
    output has to make the agreement dramatically worse, which it only does
    if all three are genuinely on one convention now.
    """
    solver = request.getfixturevalue(backend)
    assert _relative(np.conj(solver.pressure("onaxis")),
                     analytic["onaxis"]) > 1.0


# --- interchangeability ---------------------------------------------------

@needs_numcalc
def test_both_backends_shape_their_results_the_same(model, numcalc, bempp):
    """Same Model in, same set names and array shapes out — which is what
    lets anything downstream of Results take either backend."""
    assert bempp.set_names == numcalc.set_names
    np.testing.assert_array_equal(bempp.frequencies, numcalc.frequencies)
    for name in numcalc.set_names:
        assert bempp.pressure(name).shape == numcalc.pressure(name).shape
        assert bempp.velocity(name).shape == numcalc.velocity(name).shape
        np.testing.assert_array_equal(bempp.offsets[name],
                                      numcalc.offsets[name])
    assert (bempp.boundary_pressure().shape
            == numcalc.boundary_pressure().shape)
    np.testing.assert_allclose(bempp.boundary_velocity(),
                               numcalc.boundary_velocity())


@needs_numcalc
def test_both_backends_leave_the_same_bins_uncomputed(model, numcalc, bempp):
    """``bins`` means the same thing to both, down to which bins are NaN."""
    uncomputed = np.setdiff1d(np.arange(model.n_frequencies), COMPUTED)
    assert uncomputed.size, "every bin was solved — this test proves nothing"
    for solver in (numcalc, bempp):
        p = solver.pressure("onaxis")
        assert np.all(np.isfinite(p[:, COMPUTED]))
        assert np.all(np.isnan(p[:, uncomputed]))
