"""Frequency plots must show which bins were actually computed.

A partial-band run leaves isolated computed bins with no line-segment
neighbour, so every frequency-domain plot needs markers when the band has
gaps — not just ``freq_magnitude``. This file covers all four:
``freq_magnitude``, ``freq_phase``, ``angle_response`` (circular sets),
``onaxis_response`` (on-axis sets).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bemkit.analysis.result import BEMResult
from bemkit.sim.fields import make_fields_onaxis, make_fields_polar
from bemviz.viz.freq import angle_response, freq_magnitude, freq_phase, onaxis_response


def _make_result(points, missing=(), nf=4, name="t"):
    n = len(points)
    p = np.ones((n, nf), dtype=complex)
    for b in missing:
        p[:, b - 1] = np.nan
    return BEMResult(
        name=name,
        frequencies=np.array([250.0, 500.0, 750.0, 1000.0])[:nf],
        eval_points=points,
        pressure_raw=p, pressure_acc=p,
        velocity_raw=p, velocity_acc=p,
        rho0=1.2, speed_of_sound=340.0,
    )


def _result(missing=()):
    return _make_result(np.zeros((2, 3)), missing=missing)


def _circular_result(missing=()):
    """A circular set whose 30° spacing covers the default off-axis fan
    (0°, ±30°, ±60°, ±90°) exactly, so angle_response has curves to draw."""
    return _make_result(make_fields_polar(dims="zx", res=30.0, radius=1.0),
                        missing=missing)


def _onaxis_result(missing=()):
    return _make_result(make_fields_onaxis("z", [0.1, 0.5, 1.0, 2.0]),
                        missing=missing)


def test_full_band_draws_a_plain_line():
    fig, ax = plt.subplots()
    lines = freq_magnitude(ax, _result())
    assert all(ln.get_marker() in (None, "None", "") for ln in lines)
    plt.close(fig)


def test_sparse_band_marks_the_computed_bins():
    fig, ax = plt.subplots()
    lines = freq_magnitude(ax, _result(missing=(2, 3)))
    assert all(ln.get_marker() == "o" for ln in lines)
    plt.close(fig)


def test_an_explicit_marker_still_wins():
    fig, ax = plt.subplots()
    lines = freq_magnitude(ax, _result(missing=(2,)), marker="s")
    assert all(ln.get_marker() == "s" for ln in lines)
    plt.close(fig)


class TestFreqPhaseSparse:
    def test_full_band_draws_a_plain_line(self):
        fig, ax = plt.subplots()
        lines = freq_phase(ax, _result())
        assert all(ln.get_marker() in (None, "None", "") for ln in lines)
        plt.close(fig)

    def test_sparse_band_marks_the_computed_bins(self):
        fig, ax = plt.subplots()
        lines = freq_phase(ax, _result(missing=(2, 3)))
        assert all(ln.get_marker() == "o" for ln in lines)
        plt.close(fig)

    def test_an_explicit_marker_still_wins(self):
        fig, ax = plt.subplots()
        lines = freq_phase(ax, _result(missing=(2,)), marker="s")
        assert all(ln.get_marker() == "s" for ln in lines)
        plt.close(fig)


class TestAngleResponseSparse:
    def test_full_band_draws_a_plain_line(self):
        fig, ax = plt.subplots()
        lines = angle_response(ax, _circular_result())
        assert all(ln.get_marker() in (None, "None", "") for ln in lines)
        plt.close(fig)

    def test_sparse_band_marks_the_computed_bins(self):
        fig, ax = plt.subplots()
        lines = angle_response(ax, _circular_result(missing=(2, 3)))
        assert len(lines) > 0
        assert all(ln.get_marker() == "o" for ln in lines)
        plt.close(fig)

    def test_an_explicit_marker_still_wins(self):
        fig, ax = plt.subplots()
        lines = angle_response(ax, _circular_result(missing=(2,)), marker="s")
        assert all(ln.get_marker() == "s" for ln in lines)
        plt.close(fig)


class TestOnaxisResponseSparse:
    def test_full_band_draws_a_plain_line(self):
        fig, ax = plt.subplots()
        lines = onaxis_response(ax, _onaxis_result())
        assert all(ln.get_marker() in (None, "None", "") for ln in lines)
        plt.close(fig)

    def test_sparse_band_marks_the_computed_bins(self):
        fig, ax = plt.subplots()
        lines = onaxis_response(ax, _onaxis_result(missing=(2, 3)))
        assert len(lines) > 0
        assert all(ln.get_marker() == "o" for ln in lines)
        plt.close(fig)

    def test_an_explicit_marker_still_wins(self):
        fig, ax = plt.subplots()
        lines = onaxis_response(ax, _onaxis_result(missing=(2,)), marker="s")
        assert all(ln.get_marker() == "s" for ln in lines)
        plt.close(fig)
