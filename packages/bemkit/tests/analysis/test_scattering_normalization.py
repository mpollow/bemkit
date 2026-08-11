"""Normalization semantics for models with no velocity drive (Tier 3).

``acc`` divides by the vibrating surface area, so it only means anything for a
velocity-driven model. A scattering model — a plane wave or point source against
a rigid or absorbing surface — has no such area. The toolkit used to warn and
hand back the *raw* array, which produced raw Pascals wearing an
acceleration-normalized label: wrong by a frequency-dependent factor, and
invisible on a plot.

Nothing in the suite exercised a scattering bundle before, which is how that
survived.
"""

import numpy as np
import pytest

from bemkit.analysis.normalization import NoVelocityDrive
from bemkit.analysis.simulation import SimulationData
from pynumcalc.deck import BC, Deck, Source
from pynumcalc.readers import read_nc_inp_boundary_conditions

CONFIG = {"sr": 8000, "nbins": 3, "air_density": 1.2, "speed_of_sound": 340.0}


def _bundle(tmp_path, boundary, plane_waves=()):
    """A minimal solved bundle: two triangles, two bins, given BCs."""
    nc = tmp_path / "numcalc" / "v"
    (nc / "ObjectMesh").mkdir(parents=True)
    (nc / "EvalGrid").mkdir(parents=True)
    (nc / "ObjectMesh" / "Nodes.txt").write_text(
        "4\n     0 0 0 0\n     1 1 0 0\n     2 0 1 0\n     3 1 1 0\n")
    (nc / "ObjectMesh" / "Elements.txt").write_text(
        "2\n     0      0      1      2 0 0 0\n     1      1      3      2 0 0 0\n")
    (nc / "EvalGrid" / "Nodes.txt").write_text("1\n200000 0 0 1\n")
    (nc / "EvalGrid" / "Elements.txt").write_text("1\n200000 200000 200000 200000 2 0 1\n")

    Deck(title="t", frequencies=np.array([2000.0, 4000.0]),
         n_total_nodes=5, n_total_elements=3,
         boundary=list(boundary),
         plane_waves=list(plane_waves)).write(nc / "NC.inp")

    head = "header1\nheader2\nheader3\n"
    for b in (1, 2):
        d = nc / "be.out" / f"be.{b}"
        d.mkdir(parents=True)
        # be.out files carry three header lines before the data rows.
        for q, n in (("pBoundary", 2), ("pEvalGrid", 1), ("vBoundary", 2)):
            d.joinpath(q).write_text(
                head + "".join(f"{i} {b}.0 0.5\n" for i in range(n)))
        d.joinpath("vEvalGrid").write_text(
            head + f"0 {b}.0 0.0 {b}.0 0.0 {b}.0 0.0\n")
    return nc


class TestAVibCorrectness:
    """Two ways A_vib silently drifted from the truth."""

    def test_overlapping_ranges_count_an_element_once(self, tmp_path):
        # A [BOUNDARY] entry may cover elements the vibrating group already
        # drove; both emit a VELO line over the same triangle.
        nc = _bundle(tmp_path, [BC(0, 1, "VELO", 1.0), BC(1, 1, "VELO", 1.0)])
        vib = read_nc_inp_boundary_conditions(nc / "NC.inp")["vibrating"]
        np.testing.assert_array_equal(vib, [0, 1])

        both = SimulationData(nc, CONFIG).A_vib
        one = SimulationData(
            _bundle(tmp_path / "b", [BC(0, 1, "VELO", 1.0)]), CONFIG).A_vib
        assert both == pytest.approx(one)

    def test_zero_velocity_is_rigid_not_driven(self, tmp_path):
        """v0 = 0 is exactly how NumCalc spells sound-hard."""
        nc = _bundle(tmp_path, [BC(0, 0, "VELO", 1.0), BC(1, 1, "VELO", 0.0)])
        vib = read_nc_inp_boundary_conditions(nc / "NC.inp")["vibrating"]
        np.testing.assert_array_equal(vib, [0])   # not [0, 1]

    def test_non_velocity_conditions_are_not_sources(self, tmp_path):
        nc = _bundle(tmp_path, [BC(0, 0, "VELO", 1.0),
                                BC(1, 1, "ADMI", 0.4, imag=-0.2)])
        vib = read_nc_inp_boundary_conditions(nc / "NC.inp")["vibrating"]
        np.testing.assert_array_equal(vib, [0])


class TestScatteringModel:
    """A plane wave against an entirely rigid surface."""

    @pytest.fixture
    def sim(self, tmp_path):
        nc = _bundle(tmp_path, boundary=[],
                     plane_waves=[Source(1, (0.0, 0.0, 1.0), 1.0)])
        return SimulationData(nc, CONFIG)

    def test_has_no_vibrating_area(self, sim):
        assert sim.A_vib is None
        assert sim.is_velocity_driven is False

    def test_acc_raises_instead_of_returning_raw(self, sim):
        with pytest.raises(NoVelocityDrive, match="norm='raw'"):
            sim.pressure(None, norm="acc")
        with pytest.raises(NoVelocityDrive):
            sim.velocity(None, norm="acc")
        with pytest.raises(NoVelocityDrive):
            sim.boundary_pressure(norm="acc")

    def test_single_bin_accessors_refuse_too(self, sim):
        """pressure_at used to skip normalization silently when A_vib was None,
        so the 3D viewer showed raw data on an acc-labelled colour bar."""
        with pytest.raises(NoVelocityDrive):
            sim.pressure_at(1, domain="eval", norm="acc")
        with pytest.raises(NoVelocityDrive):
            sim.velocity_at(1, domain="eval", norm="acc")

    def test_auto_resolves_to_raw(self, sim):
        assert sim.effective_norm("auto") == "raw"
        np.testing.assert_array_equal(sim.pressure(None, norm="auto"),
                                      sim.pressure(None, norm="raw"))

    def test_raw_is_unaffected(self, sim):
        p = sim.pressure(None, norm="raw")
        assert np.isfinite(p).all() and np.abs(p).max() > 0

    def test_result_carries_no_acc_but_still_works(self, sim):
        r = sim.result(None)
        assert r.pressure_acc is None and r.velocity_acc is None
        assert r.has_acc is False
        assert r.effective_norm("auto") == "raw"
        np.testing.assert_array_equal(r.data("pressure", "auto"), r.pressure_raw)

    def test_result_acc_request_names_the_way_out(self, sim):
        with pytest.raises(NoVelocityDrive, match="norm='raw'"):
            sim.result(None).data("pressure", "acc")

    def test_consumers_resolve_auto_to_raw(self, sim):
        """What a display layer must key off: "auto" reports *raw* here, so
        nothing can label a scattering result in acceleration-normalized units.

        (This used to assert on a matplotlib axis label, until the 2D plotting
        family was removed on 2026-08-12.)"""
        assert sim.result(None).effective_norm("auto") == "raw"
        assert sim.effective_norm("auto") == "raw"


class TestVelocityDrivenModelUnchanged:
    """The normal case must behave exactly as before."""

    @pytest.fixture
    def sim(self, tmp_path):
        return SimulationData(
            _bundle(tmp_path, [BC(0, 1, "VELO", 1.0)]), CONFIG)

    def test_auto_resolves_to_acc(self, sim):
        assert sim.is_velocity_driven is True
        assert sim.effective_norm("auto") == "acc"
        np.testing.assert_allclose(sim.pressure(None, norm="auto"),
                                   sim.pressure(None, norm="acc"))

    def test_result_has_acc(self, sim):
        r = sim.result(None)
        assert r.has_acc and r.effective_norm("auto") == "acc"

    def test_consumers_resolve_auto_to_acc(self, sim):
        assert sim.result(None).effective_norm("auto") == "acc"
        assert sim.effective_norm("auto") == "acc"


def test_unknown_norm_is_rejected(tmp_path):
    sim = SimulationData(_bundle(tmp_path, [BC(0, 1, "VELO", 1.0)]), CONFIG)
    with pytest.raises(ValueError, match="must be 'auto', 'acc' or 'raw'"):
        sim.effective_norm("normalised")


def test_subset_of_a_scattering_result_stays_acc_free(tmp_path):
    """`result[idx]` sliced pressure_acc unconditionally."""
    sim = SimulationData(
        _bundle(tmp_path, boundary=[],
                plane_waves=[Source(1, (0.0, 0.0, 1.0), 1.0)]), CONFIG)
    sub = sim.result(None)[[0]]
    assert sub.pressure_acc is None and sub.has_acc is False
    assert sub.pressure_raw.shape[0] == 1


def test_missing_bins_works_without_acc(tmp_path):
    """missing_bins read pressure_acc, which a scattering model does not have."""
    sim = SimulationData(
        _bundle(tmp_path, boundary=[],
                plane_waves=[Source(1, (0.0, 0.0, 1.0), 1.0)]), CONFIG)
    assert sim.result(None).missing_bins == []


class TestNonUniformBand:
    """A [frequencies] model is fine in the frequency domain and meaningless
    in the time domain (Tier 7)."""

    def _result(self, tmp_path, freqs):
        import numpy as np
        from bemkit.analysis.result import BEMResult
        n = len(freqs)
        return BEMResult(name="t", frequencies=np.asarray(freqs, float),
                         eval_points=np.zeros((1, 3)),
                         pressure_raw=np.ones((1, n), complex),
                         pressure_acc=np.ones((1, n), complex),
                         velocity_raw=np.ones((1, n)),
                         velocity_acc=np.ones((1, n)),
                         rho0=1.2, speed_of_sound=340.0)

    def test_uniform_band_is_recognised(self, tmp_path):
        assert self._result(tmp_path, [250., 500., 750., 1000.]).is_uniform_band

    def test_log_band_is_not(self, tmp_path):
        assert not self._result(tmp_path, [200., 400., 800., 1600.]).is_uniform_band

    def test_evenly_spaced_but_offset_band_is_not(self, tmp_path):
        """An FFT grid starts at its own step, so 150/250/350 is evenly spaced
        and still not a grid an IRFFT can read."""
        assert not self._result(tmp_path, [150., 250., 350.]).is_uniform_band

    def test_impulse_response_refuses_a_non_uniform_band(self, tmp_path):
        from bemkit.analysis.result import IncompleteBand
        with pytest.raises(IncompleteBand, match="uniform frequency grid"):
            self._result(tmp_path, [200., 400., 800., 1600.]).impulse_response()

    def test_impulse_response_still_works_on_the_fft_grid(self, tmp_path):
        t, ir = self._result(tmp_path, [250., 500., 750., 1000.]).impulse_response()
        assert ir.shape[0] == 1 and len(t) == ir.shape[1]
