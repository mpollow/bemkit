"""Tests for bemkit.analysis.result — BEMResult accessors, slicing, properties."""

import numpy as np
import pytest

from bemkit.analysis.result import BEMResult


@pytest.fixture()
def simple_result():
    """A BEMResult with 4 points, 3 frequencies, known complex pressure."""
    freqs = np.array([100.0, 200.0, 300.0])
    pts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [float('nan'), float('nan'), float('nan')],  # dropped point
    ])
    # pressure: p[i,f] = (i+1) + f*1j for simple verification
    pressure = np.array([
        [1+100j, 1+200j, 1+300j],
        [2+100j, 2+200j, 2+300j],
        [3+100j, 3+200j, 3+300j],
        [np.nan+0j, np.nan+0j, np.nan+0j],
    ])
    return BEMResult(
        name="test",
        frequencies=freqs,
        eval_points=pts,
        pressure_raw=pressure,
        pressure_acc=pressure * 2,  # acc = 2x raw for easy checking
        velocity_raw=np.zeros_like(pressure),
        velocity_acc=np.zeros_like(pressure),
        rho0=1.2,
        speed_of_sound=340.0,
    )


class TestBEMResultProperties:
    def test_n_points(self, simple_result):
        assert simple_result.n_points == 4

    def test_n_valid(self, simple_result):
        assert simple_result.n_valid == 3

    def test_n_freq(self, simple_result):
        assert simple_result.n_freq == 3

    def test_valid_mask(self, simple_result):
        mask = simple_result.valid_mask
        np.testing.assert_array_equal(mask, [True, True, True, False])

    def test_xyz(self, simple_result):
        assert simple_result.x[0] == 1.0
        assert simple_result.y[1] == 1.0
        assert simple_result.z[2] == 1.0

    def test_spherical_coords(self, simple_result):
        r = simple_result.r
        assert r[0] == pytest.approx(1.0)
        assert r[1] == pytest.approx(1.0)

    def test_repr(self, simple_result):
        s = repr(simple_result)
        assert "test" in s
        assert "3/4" in s
        assert "3 freqs" in s


class TestBEMResultData:
    def test_data_raw(self, simple_result):
        d = simple_result.data("pressure", "raw")
        assert d[0, 0] == pytest.approx(1+100j)

    def test_data_acc(self, simple_result):
        d = simple_result.data("pressure", "acc")
        assert d[0, 0] == pytest.approx(2+200j)  # 2x raw

    def test_magnitude(self, simple_result):
        mag = simple_result.magnitude("pressure", "raw")
        assert mag[0, 0] == pytest.approx(abs(1+100j))

    def test_magnitude_db(self, simple_result):
        db = simple_result.magnitude_db("pressure", "raw")
        expected = 20 * np.log10(abs(1+100j))
        assert db[0, 0] == pytest.approx(expected)

    def test_phase(self, simple_result):
        ph = simple_result.phase("pressure", "raw")
        expected = np.angle(1+100j)
        assert ph[0, 0] == pytest.approx(expected)

    def test_phase_unwrap(self, simple_result):
        ph = simple_result.phase("pressure", "raw", unwrap=True)
        assert ph.shape == (4, 3)

    def test_invalid_quantity_raises(self, simple_result):
        with pytest.raises(ValueError, match="Unknown"):
            simple_result.data("invalid", "raw")


class TestBEMResultFreqHelpers:
    def test_at_freq_exact(self, simple_result):
        assert simple_result.at_freq(200) == 1

    def test_at_freq_nearest(self, simple_result):
        assert simple_result.at_freq(190) == 1

    def test_freq_slice(self, simple_result):
        s = simple_result.freq_slice(100, 200)
        assert s == slice(0, 2)

    def test_freq_slice_unbounded(self, simple_result):
        s = simple_result.freq_slice(f_min=None, f_max=None)
        assert s == slice(0, 3)


class TestBEMResultSlicing:
    def test_int_index(self, simple_result):
        r = simple_result[0]
        assert r.n_points == 1
        np.testing.assert_allclose(r.eval_points, [[1, 0, 0]])

    def test_slice_index(self, simple_result):
        r = simple_result[:2]
        assert r.n_points == 2

    def test_bool_mask(self, simple_result):
        mask = simple_result.valid_mask
        r = simple_result[mask]
        assert r.n_points == 3
        assert r.n_valid == 3

    def test_preserves_metadata(self, simple_result):
        r = simple_result[0]
        assert r.name == "test"
        assert r.rho0 == 1.2
        assert r.speed_of_sound == 340.0
        np.testing.assert_array_equal(r.frequencies, simple_result.frequencies)


class TestBEMResultImpulseResponse:
    def test_ir_shape(self, simple_result):
        time_s, ir = simple_result.impulse_response()
        assert ir.shape[0] == 4  # n_points
        assert len(time_s) == ir.shape[1]

    def test_ir_time_starts_at_zero(self, simple_result):
        time_s, _ = simple_result.impulse_response()
        assert time_s[0] == 0.0


class TestMissingBins:
    """A partial-band run leaves whole frequency columns NaN."""

    def _result(self, missing=()):
        """A 3-point, 4-bin result with the given 1-based bins blanked."""
        from bemkit.analysis.result import BEMResult

        nf, n = 4, 3
        p = np.ones((n, nf), dtype=complex)
        for b in missing:
            p[:, b - 1] = np.nan
        return BEMResult(
            name="t",
            frequencies=np.array([250.0, 500.0, 750.0, 1000.0]),
            eval_points=np.zeros((n, 3)),
            pressure_raw=p, pressure_acc=p,
            velocity_raw=p, velocity_acc=p,
            rho0=1.2, speed_of_sound=340.0,
        )

    def test_full_band_has_none_missing(self):
        assert self._result().missing_bins == []

    def test_blank_columns_are_reported_1_based(self):
        assert self._result(missing=(2, 4)).missing_bins == [2, 4]

    def test_dropped_points_are_not_mistaken_for_missing_bins(self):
        """Fieldpoint validation NaNs whole rows; that is not a missing bin."""
        r = self._result()
        r.eval_points[1] = np.nan
        r.pressure_acc[1, :] = np.nan
        assert r.missing_bins == []

    def test_impulse_response_refuses_a_sparse_band(self):
        from bemkit.analysis.result import IncompleteBand

        with pytest.raises(IncompleteBand, match="2 of 4"):
            self._result(missing=(2, 4)).impulse_response()

    def test_impulse_response_still_works_on_a_full_band(self):
        time_s, ir = self._result().impulse_response()
        assert ir.shape[0] == 3 and len(time_s) == ir.shape[1]

    def test_incomplete_band_is_a_value_error(self):
        """cli/_plot.py catches it alongside NotCircular/NotOnAxis."""
        from bemkit.analysis.result import IncompleteBand

        assert issubclass(IncompleteBand, ValueError)
