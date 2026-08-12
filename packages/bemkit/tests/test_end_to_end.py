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

from bemkit.study import load_study
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


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """One real bundle, solved for a single bin, shared by the class below."""
    wd = tmp_path_factory.mktemp("coarse")
    model = load_study(STUDY)["coarse"]
    NumCalcSolver().solve(model, workdir=wd, bins=[1])
    return wd, model


class TestTheViewerReadsWhatWasSolved:
    """`bem STUDY results` reads geometry off disk and numbers out of
    Results. This pins that the two line up on a real bundle — the failure
    mode otherwise is a picture that renders perfectly and is wrong.
    """

    def test_geometry_comes_back_matching_the_model_mesh(self, solved):
        from bemkit.view.bundle import Bundle
        wd, model = solved
        b = Bundle(wd, model)
        np.testing.assert_allclose(b.nodes, model.mesh.vertices, atol=1e-9)
        np.testing.assert_array_equal(b.elements, model.mesh.triangles)

    def test_only_the_one_solved_bin_is_offered(self, solved):
        """--preview / a single -bins run leaves 63 NaN bins. The slider must
        offer the one that exists, and nothing else."""
        from bemkit.view.bundle import Bundle
        wd, model = solved
        np.testing.assert_array_equal(Bundle(wd, model).bins, [0])

    def test_points_and_pressure_share_an_index(self, solved):
        """Both are split by the same offsets, so row i of one is point i of
        the other. Getting this wrong colours the polar circle with the
        on-axis values and looks entirely plausible."""
        from bemkit.view.bundle import Bundle
        wd, model = solved
        b = Bundle(wd, model)
        assert set(b.set_names) == {"onaxis", "polar"}
        for name in b.set_names:
            assert len(b.points(name)) == len(b.pressure(0, name))
        # onaxis is +z at increasing radius, and its pressure falls off.
        pts = b.points("onaxis")
        assert np.all(pts[:, 2] > 0)
        assert np.all(np.diff(pts[:, 2]) > 0)
        mag = np.abs(b.pressure(0, "onaxis"))
        assert mag[0] > mag[-1]

    def test_acc_normalization_divides_by_j_rho_omega_area(self, solved):
        from bemkit.view.bundle import Bundle
        wd, model = solved
        raw = Bundle(wd, model, norm="raw").pressure(0, "onaxis")
        acc = Bundle(wd, model, norm="acc").pressure(0, "onaxis")
        omega = 2 * np.pi * model.frequencies[0]
        factor = 1j * model.medium.rho * omega * model.vibrating_area
        np.testing.assert_allclose(acc, raw / factor, rtol=1e-9)

    def test_the_cap_area_is_a_small_fraction_of_the_sphere(self, solved):
        """A 20-degree cap on a 50 mm sphere is 2*pi*r^2*(1-cos 20 deg) =
        9.48e-4 m^2, about 3 % of the 4*pi*r^2 total. The tolerance is loose
        because a faceted mesh chords inside the sphere and reads a few
        percent low; what this pins is the order of magnitude, and that
        A_vib is the cap and not the whole surface (which is what a group
        resolving to everything, or a VELO 0 range being counted, would
        give).
        """
        _wd, model = solved
        exact = 2 * np.pi * 0.05 ** 2 * (1 - np.cos(np.radians(20.0)))
        assert model.vibrating_area == pytest.approx(exact, rel=0.10)
        total = float(model.mesh.areas.sum())
        assert model.vibrating_area / total == pytest.approx(0.030, rel=0.15)
