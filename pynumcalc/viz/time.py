"""Time-domain plotting functions."""

import numpy as np

from pynumcalc.viz._style import (
    point_colors, make_point_label, TIME_LABEL_MS, TIME_LABEL_S, IR_LABEL,
)
from pynumcalc.viz.freq import _resolve_points

# Time unit scale factors
_TIME_SCALES = {"s": 1.0, "ms": 1e3, "samples": None}


def impulse_response(ax, result, *, points=None, quantity="pressure", norm="acc",
                     time_range=None, time_unit="ms",
                     label_fmt="z={z:.2f} m", **kwargs):
    """Plot impulse response waveforms (IRFFT).

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    points : None, int, array, slice, or bool mask
        Which evaluation points to plot. None = all valid.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    time_range : (t_min, t_max) or None
        Time range in the selected unit.
    time_unit : "ms", "s", or "samples"
    label_fmt : str or callable
        Label format (see ``make_point_label``).
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    list of Line2D
    """
    indices = _resolve_points(result, points)
    colors = point_colors(len(indices))

    time_s, ir = result.impulse_response(quantity, norm)

    if time_unit == "samples":
        t = np.arange(ir.shape[1])
        xlabel = "Samples"
    else:
        scale = _TIME_SCALES[time_unit]
        t = time_s * scale
        xlabel = TIME_LABEL_MS if time_unit == "ms" else TIME_LABEL_S

    lines = []
    for ci, idx in enumerate(indices):
        if np.any(np.isnan(ir[idx])):
            continue
        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = make_point_label(label_fmt, idx, result.eval_points[idx])
        ln = ax.plot(t, ir[idx], label=label, **kw)
        lines.extend(ln)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(IR_LABEL)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if time_range is not None:
        ax.set_xlim(time_range)

    return lines


def energy_time_curve(ax, result, *, points=None, quantity="pressure", norm="acc",
                      db=True, time_range=None, time_unit="ms",
                      label_fmt="z={z:.2f} m", **kwargs):
    """Plot cumulative energy (Schroeder integral).

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    points, quantity, norm, time_range, time_unit, label_fmt, **kwargs
        See ``impulse_response``.
    db : bool
        If True, plot in dB (normalized to max).

    Returns
    -------
    list of Line2D
    """
    indices = _resolve_points(result, points)
    colors = point_colors(len(indices))

    time_s, ir = result.impulse_response(quantity, norm)

    if time_unit == "samples":
        t = np.arange(ir.shape[1])
        xlabel = "Samples"
    else:
        scale = _TIME_SCALES[time_unit]
        t = time_s * scale
        xlabel = TIME_LABEL_MS if time_unit == "ms" else TIME_LABEL_S

    lines = []
    for ci, idx in enumerate(indices):
        if np.any(np.isnan(ir[idx])):
            continue

        # Schroeder integral: reverse cumulative sum of squared IR
        energy = np.cumsum(ir[idx][::-1]**2)[::-1]

        if db:
            with np.errstate(divide="ignore"):
                vals = 10 * np.log10(energy / energy.max())
        else:
            vals = energy

        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = make_point_label(label_fmt, idx, result.eval_points[idx])
        ln = ax.plot(t, vals, label=label, **kw)
        lines.extend(ln)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy (dB)" if db else "Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if time_range is not None:
        ax.set_xlim(time_range)

    return lines
