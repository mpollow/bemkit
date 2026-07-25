"""Frequency plots must show which bins were actually computed."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bemkit.analysis.result import BEMResult
from bemkit.viz.freq import freq_magnitude


def _result(missing=()):
    nf, n = 4, 2
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
