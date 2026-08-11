"""Tests for the dataset-tailored plots: isobaric map, off-axis fan, on-axis.

Only the data the plots put on the axes is checked — what the figure looks
like is not asserted on, for the reason given in tests/viz/test_symmetry.py.
"""

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bemkit.analysis.result import BEMResult  # noqa: E402
from bemkit.sim.fields import make_fields_onaxis, make_fields_polar  # noqa: E402
from bemviz.viz._angles import NotCircular, NotOnAxis, circle_angles  # noqa: E402
from bemviz.viz.directivity import (  # noqa: E402
    _contour_levels, directivity_levels, isobaric, level_label,
)
from bemviz.viz.freq import angle_response, onaxis_response  # noqa: E402
from bemviz.viz.time import angle_impulse_response  # noqa: E402


@pytest.fixture()
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def make_result(points, pressure, *, freqs, name="test"):
    pressure = np.asarray(pressure, dtype=complex)
    return BEMResult(
        name=name,
        frequencies=np.asarray(freqs, dtype=float),
        eval_points=np.asarray(points, dtype=float),
        pressure_raw=pressure,
        pressure_acc=pressure,
        velocity_raw=np.zeros_like(pressure),
        velocity_acc=np.zeros_like(pressure),
        rho0=1.2,
        speed_of_sound=340.0,
    )


@pytest.fixture()
def beaming_circle():
    """A circular set whose beam narrows with frequency.

    |p|(θ, f) = exp(-(θ/beamwidth(f))²) with beamwidth shrinking from 90° at
    the lowest frequency to 20° at the highest — an idealized but honest
    stand-in for a piston, with an exactly known 0° reference.
    """
    freqs = np.array([250.0, 1000.0, 4000.0, 16000.0])
    points = make_fields_polar(dims="zx", res=5.0, radius=1.0)
    angles, _, _ = circle_angles(make_result(points,
                                             np.ones((len(points), len(freqs))),
                                             freqs=freqs))
    beamwidth = np.array([90.0, 60.0, 35.0, 20.0])
    mag = np.exp(-(angles[:, None] / beamwidth[None, :]) ** 2)
    return make_result(points, mag.astype(complex), freqs=freqs, name="polar")


@pytest.fixture()
def decaying_onaxis():
    """An on-axis set with exact 1/r free-field decay and a 1 kHz bump."""
    freqs = np.array([250.0, 500.0, 1000.0, 2000.0])
    radii = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    points = make_fields_onaxis("z", radii)
    source = np.array([1.0, 1.0, 3.0, 1.0])  # +9.5 dB at 1 kHz
    mag = source[None, :] / radii[:, None]
    return make_result(points, mag.astype(complex), freqs=freqs, name="onaxis")


class TestDirectivityLevels:
    def test_onaxis_reference_flattens_zero_degrees(self, beaming_circle):
        angles, _, _ = circle_angles(beaming_circle)
        order = np.argsort(angles)
        levels = directivity_levels(
            beaming_circle.magnitude()[order], angles[order], "onaxis")

        on = int(np.argmin(np.abs(angles[order])))
        assert levels[on] == pytest.approx(np.zeros(beaming_circle.n_freq))
        # Everything off axis is at or below the reference for this source.
        assert levels.max() == pytest.approx(0.0)

    def test_onaxis_reference_is_per_frequency(self, beaming_circle):
        angles, _, _ = circle_angles(beaming_circle)
        order = np.argsort(angles)
        levels = directivity_levels(
            beaming_circle.magnitude()[order], angles[order], "onaxis")

        at30 = int(np.argmin(np.abs(angles[order] - 30.0)))
        # The beam narrows with frequency, so 30° falls away monotonically.
        assert np.all(np.diff(levels[at30]) < 0)

    def test_max_reference_peaks_at_zero_db(self, beaming_circle):
        angles, _, _ = circle_angles(beaming_circle)
        levels = directivity_levels(beaming_circle.magnitude(), angles, "max")
        assert np.nanmax(levels, axis=0) == pytest.approx(np.zeros(4))

    def test_globalmax_reference_uses_one_value(self):
        mag = np.array([[1.0, 2.0], [0.5, 1.0]])
        levels = directivity_levels(mag, np.array([0.0, 90.0]), "globalmax")
        assert levels.max() == pytest.approx(0.0)
        assert levels[0, 0] == pytest.approx(20 * np.log10(0.5))

    def test_numeric_reference_is_absolute(self):
        mag = np.array([[10.0], [1.0]])
        levels = directivity_levels(mag, np.array([0.0, 90.0]), 1.0)
        assert levels[:, 0] == pytest.approx([20.0, 0.0])

    def test_labels_name_the_reference(self):
        assert "on-axis" in level_label("onaxis")
        assert "peak" in level_label("max")
        assert "1 Pa" in level_label(1.0)

    def test_max_reference_does_not_warn_on_an_all_nan_frequency(self):
        # A partial-band run leaves whole frequency columns all-NaN; "max"
        # takes nanmax over angle per frequency, which used to raise a raw
        # RuntimeWarning ("All-NaN slice encountered") straight to the user.
        mag = np.array([[1.0, np.nan], [0.5, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            levels = directivity_levels(mag, np.array([0.0, 90.0]), "max")
        assert np.isnan(levels[:, 1]).all()

    def test_globalmax_reference_does_not_warn_when_entirely_nan(self):
        mag = np.full((2, 2), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            levels = directivity_levels(mag, np.array([0.0, 90.0]), "globalmax")
        assert np.isnan(levels).all()


class TestContourLevels:
    def test_step_fills_the_color_range(self):
        levels = _contour_levels(6.0, None, (-30.0, 0.0))
        assert list(levels) == pytest.approx([-30, -24, -18, -12, -6])

    def test_explicit_levels_pass_through(self):
        levels = _contour_levels([-6.0, -3.0], None, (-30.0, 0.0))
        assert list(levels) == pytest.approx([-6.0, -3.0])


class TestIsobaric:
    def test_mesh_covers_angles_and_frequencies(self, ax, beaming_circle):
        mesh = isobaric(ax, beaming_circle)
        assert mesh.get_array().shape == (72, 4)

        lo, hi = ax.get_ylim()
        assert lo == pytest.approx(-180.0)
        assert hi < 180.0
        assert ax.get_xlim() == pytest.approx((250.0, 16000.0))

    def test_log_frequency_axis_by_default(self, ax, beaming_circle):
        isobaric(ax, beaming_circle)
        assert ax.get_xscale() == "log"

        fig2, ax2 = plt.subplots()
        isobaric(ax2, beaming_circle, log_freq=False)
        assert ax2.get_xscale() == "linear"
        plt.close(fig2)

    def test_color_limits_and_reference(self, ax, beaming_circle):
        mesh = isobaric(ax, beaming_circle, clim=(-24.0, 0.0))
        assert mesh.get_clim() == pytest.approx((-24.0, 0.0))
        # On-axis reference: the 0° row is exactly 0 dB.
        levels = np.asarray(mesh.get_array())
        on_axis_row = 36  # 72 directions sorted from -180°, so 0° is row 36
        assert levels[on_axis_row] == pytest.approx(np.zeros(4))

    def test_frequency_range_is_honored(self, ax, beaming_circle):
        mesh = isobaric(ax, beaming_circle, freq_range=(1000.0, 4000.0))
        assert mesh.get_array().shape == (72, 2)

    def test_contours_draw_lines(self, ax, beaming_circle):
        isobaric(ax, beaming_circle, contours=6.0)
        assert len(ax.collections) > 1

    def test_rejects_a_non_circular_set(self, ax, decaying_onaxis):
        with pytest.raises(NotCircular):
            isobaric(ax, decaying_onaxis)

    def test_dropped_points_are_excluded(self, ax, beaming_circle):
        points = beaming_circle.eval_points.copy()
        points[5] = np.nan
        holed = make_result(points, beaming_circle.pressure_acc,
                            freqs=beaming_circle.frequencies)
        mesh = isobaric(ax, holed)
        assert mesh.get_array().shape == (71, 4)


class TestAngleResponse:
    def test_default_fan_is_seven_curves(self, ax, beaming_circle):
        lines = angle_response(ax, beaming_circle)
        assert len(lines) == 7
        assert [ln.get_label() for ln in lines] == [
            "+0°", "+30°", "-30°", "+60°", "-60°", "+90°", "-90°"]

    def test_pairs_share_color_and_differ_in_style(self, ax, beaming_circle):
        lines = angle_response(ax, beaming_circle)
        plus30, minus30 = lines[1], lines[2]
        assert plus30.get_color() == minus30.get_color()
        assert plus30.get_linestyle() != minus30.get_linestyle()

    def test_absolute_reference_by_default(self, ax, beaming_circle):
        lines = angle_response(ax, beaming_circle)
        # |p| = 1 on axis at every frequency, so 0 dB re 1.0.
        assert lines[0].get_ydata() == pytest.approx(np.zeros(4))

    def test_onaxis_reference_normalizes(self, ax, beaming_circle):
        lines = angle_response(ax, beaming_circle, ref="onaxis")
        assert lines[0].get_ydata() == pytest.approx(np.zeros(4))
        assert np.all(lines[1].get_ydata() < 0)

    def test_selected_angles(self, ax, beaming_circle):
        lines = angle_response(ax, beaming_circle, angles=[0, 45])
        assert [ln.get_label() for ln in lines] == ["+0°", "+45°"]

    def test_rejects_a_non_circular_set(self, ax, decaying_onaxis):
        with pytest.raises(NotCircular):
            angle_response(ax, decaying_onaxis)


class TestOnAxisResponse:
    def test_distance_compensation_collapses_free_field_decay(
            self, ax, decaying_onaxis):
        lines = onaxis_response(ax, decaying_onaxis)
        assert len(lines) == 6

        curves = np.array([ln.get_ydata() for ln in lines])
        # |p| ∝ 1/r exactly, so |p|·r is the same at every distance.
        assert curves.std(axis=0) == pytest.approx(np.zeros(4), abs=1e-9)
        # …and it is the source's own response: +9.5 dB at 1 kHz.
        assert curves[0] == pytest.approx([0.0, 0.0, 20 * np.log10(3.0), 0.0])

    def test_curves_are_sorted_and_labelled_by_distance(self, ax,
                                                        decaying_onaxis):
        lines = onaxis_response(ax, decaying_onaxis)
        assert [ln.get_label() for ln in lines] == [
            "z=+0.10 m", "z=+0.25 m", "z=+0.50 m",
            "z=+1.00 m", "z=+2.00 m", "z=+4.00 m"]

    def test_origin_point_is_skipped(self, ax):
        freqs = np.array([250.0, 500.0])
        radii = [0.0, 1.0, 2.0]
        points = make_fields_onaxis("z", radii)
        mag = np.ones((3, 2))
        lines = onaxis_response(ax, make_result(points, mag, freqs=freqs))
        assert len(lines) == 2

    def test_rejects_a_circular_set(self, ax, beaming_circle):
        with pytest.raises(NotOnAxis):
            onaxis_response(ax, beaming_circle)


@pytest.fixture()
def uniform_circle(beaming_circle):
    """The same circular set on a *uniform* FFT grid.

    beaming_circle's frequencies are log-ish ([250, 1000, 4000, 16000]) — handy
    for a directivity map, meaningless for an inverse FFT, which reads the array
    index as a bin index. Time-domain tests need the real grid.
    """
    n = beaming_circle.n_freq
    return make_result(beaming_circle.eval_points, beaming_circle.pressure_acc,
                       freqs=np.arange(1, n + 1) * 250.0, name="uniform")


class TestAngleImpulseResponse:
    def test_one_curve_per_covered_direction(self, ax, uniform_circle):
        lines = angle_impulse_response(ax, uniform_circle, angles=[0, 30, -30])
        assert [ln.get_label() for ln in lines] == ["+0°", "+30°", "-30°"]

    def test_single_direction(self, ax, uniform_circle):
        lines = angle_impulse_response(ax, uniform_circle, angles=[60])
        assert len(lines) == 1

    def test_align_shifts_time_zero_to_the_first_arrival(self, ax,
                                                        uniform_circle):
        plain = angle_impulse_response(ax, uniform_circle, angles=[0])
        t_plain = plain[0].get_xdata()
        peak = t_plain[int(np.argmax(np.abs(plain[0].get_ydata())))]

        fig2, ax2 = plt.subplots()
        aligned = angle_impulse_response(ax2, uniform_circle, angles=[0],
                                         align=True)
        assert aligned[0].get_xdata() == pytest.approx(t_plain - peak)
        plt.close(fig2)

    def test_rejects_a_non_circular_set(self, ax, decaying_onaxis):
        with pytest.raises(NotCircular):
            angle_impulse_response(ax, decaying_onaxis)
