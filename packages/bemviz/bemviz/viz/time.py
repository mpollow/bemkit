"""Time-domain plotting functions."""

import numpy as np

from bemviz.viz._angles import DEFAULT_ANGLES, angle_indices
from bemviz.viz._style import (
    point_colors, angle_styles, make_point_label,
    TIME_LABEL_MS, TIME_LABEL_S, IR_LABEL, ir_label,
)
from bemviz.viz.freq import _resolve_points

# Time unit scale factors
_TIME_SCALES = {"s": 1.0, "ms": 1e3, "samples": None}


def impulse_response(ax, result, *, points=None, quantity="pressure", norm="auto",
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
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
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
    ax.set_ylabel(ir_label(norm))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if time_range is not None:
        ax.set_xlim(time_range)

    return lines


def angle_impulse_response(ax, result, *, angles=DEFAULT_ANGLES,
                           quantity="pressure", norm="auto", align=False,
                           time_range=None, time_unit="ms", tol=None,
                           **kwargs):
    """Plot impulse responses of selected directions of a circular set.

    One direction or several, overlaid on shared axes: a single angle gives
    that direction's IR, a fan gives the comparison (a ±θ pair shares a color
    and differs in line style).  On a circular set every point sits at the
    same radius, so the curves are directly comparable without any distance
    correction — only the arrival time shifts as the source's effective
    acoustic centre moves with angle.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    angles : sequence of float
        Requested directions in degrees.  Angles the set does not cover are
        dropped.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    align : bool
        Shift the time axis so t=0 is the direct-sound arrival of the *first*
        selected direction (its peak), which is what makes the relative delays
        between directions readable.
    time_range : (t_min, t_max) or None
        Time bounds in the selected unit.
    time_unit : "ms", "s", or "samples"
    tol : float or None
        Angle-matching tolerance in degrees (None = one angular step).
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    list of Line2D

    Raises
    ------
    bemviz.viz._angles.NotCircular
        If the fieldpoint set is not a circle about the origin.
    """
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
    indices, matched = angle_indices(result, angles, tol=tol)
    if len(indices) == 0:
        raise ValueError(f"none of the angles {list(angles)} are covered by "
                         f"fieldpoint set {result.name!r}")

    time_s, ir = result.impulse_response(quantity, norm)

    if time_unit == "samples":
        t = np.arange(ir.shape[1]).astype(float)
        xlabel = "Samples"
    else:
        t = time_s * _TIME_SCALES[time_unit]
        xlabel = TIME_LABEL_MS if time_unit == "ms" else TIME_LABEL_S

    if align:
        t = t - t[int(np.argmax(np.abs(ir[indices[0]])))]

    styles = angle_styles(matched)

    lines = []
    for ci, (idx, ang) in enumerate(zip(indices, matched)):
        if np.any(np.isnan(ir[idx])):
            continue
        kw = {"alpha": 0.9}
        kw.update(styles[ci])
        kw.update(kwargs)
        lines.extend(ax.plot(t, ir[idx], label=f"{ang:+.0f}°", **kw))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ir_label(norm))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2 if len(matched) > 4 else 1)

    if time_range is not None:
        ax.set_xlim(time_range)

    return lines


def energy_time_curve(ax, result, *, points=None, quantity="pressure", norm="auto",
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
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
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
