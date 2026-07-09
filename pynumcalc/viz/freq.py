"""Frequency-domain plotting functions."""

import numpy as np

from pynumcalc.viz._style import (
    point_colors, make_point_label, format_freq_axis,
    PRESSURE_LABEL_DB_ACC, PRESSURE_LABEL, PHASE_LABEL_DEG, PHASE_LABEL_RAD,
)


def _resolve_points(result, points):
    """Resolve the ``points`` parameter into an index array.

    Parameters
    ----------
    result : BEMResult
    points : None, int, array, slice, or bool mask
        None = all valid points.

    Returns
    -------
    indices : 1-d int array
    """
    if points is None:
        return np.where(result.valid_mask)[0]
    if isinstance(points, slice):
        return np.arange(*points.indices(result.n_points))
    arr = np.atleast_1d(np.asarray(points))
    if arr.dtype == bool:
        return np.where(arr)[0]
    return arr.astype(int)


def freq_magnitude(ax, result, *, points=None, quantity="pressure", norm="acc",
                   db=False, ref=1.0, freq_range=None, label_fmt="z={z:.2f} m",
                   **kwargs):
    """Plot magnitude vs frequency.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    points : None, int, array, slice, or bool mask
        Which evaluation points to plot. None = all valid.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    db : bool
        If True, plot in dB scale. Otherwise, semilogy.
    ref : float
        Reference for dB conversion.
    freq_range : (f_min, f_max) or None
        Frequency range in Hz to plot.
    label_fmt : str or callable
        Label format string (see ``make_point_label``).
    **kwargs
        Passed to ``ax.plot()`` / ``ax.semilogy()``.

    Returns
    -------
    list of Line2D
    """
    indices = _resolve_points(result, points)
    colors = point_colors(len(indices))
    freqs = result.frequencies

    if freq_range is not None:
        fsl = result.freq_slice(*freq_range)
        freqs = freqs[fsl]
    else:
        fsl = slice(None)

    lines = []
    for ci, idx in enumerate(indices):
        if db:
            vals = result.magnitude_db(quantity, norm, ref)[idx, fsl]
        else:
            vals = result.magnitude(quantity, norm)[idx, fsl]

        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)

        label = make_point_label(label_fmt, idx, result.eval_points[idx])

        if db:
            ln = ax.plot(freqs / 1000, vals, label=label, **kw)
        else:
            ln = ax.semilogy(freqs / 1000, vals, label=label, **kw)
        lines.extend(ln)

    format_freq_axis(ax)
    ax.set_ylabel(PRESSURE_LABEL_DB_ACC if not db else PRESSURE_LABEL_DB_ACC)
    ax.set_xlim(freqs[0] / 1000, freqs[-1] / 1000)
    ax.legend(loc="best")

    return lines


def freq_phase(ax, result, *, points=None, quantity="pressure", norm="acc",
               unwrap=False, degrees=True, freq_range=None,
               label_fmt="z={z:.2f} m", **kwargs):
    """Plot phase vs frequency.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    points, quantity, norm, freq_range, label_fmt, **kwargs
        See ``freq_magnitude``.
    unwrap : bool
        Unwrap phase.
    degrees : bool
        Plot in degrees (True) or radians (False).

    Returns
    -------
    list of Line2D
    """
    indices = _resolve_points(result, points)
    colors = point_colors(len(indices))
    freqs = result.frequencies

    if freq_range is not None:
        fsl = result.freq_slice(*freq_range)
        freqs = freqs[fsl]
    else:
        fsl = slice(None)

    phase = result.phase(quantity, norm, unwrap=unwrap)

    lines = []
    for ci, idx in enumerate(indices):
        vals = phase[idx, fsl]
        if degrees:
            vals = np.degrees(vals)

        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = make_point_label(label_fmt, idx, result.eval_points[idx])
        ln = ax.plot(freqs / 1000, vals, label=label, **kw)
        lines.extend(ln)

    format_freq_axis(ax)
    ax.set_ylabel(PHASE_LABEL_DEG if degrees else PHASE_LABEL_RAD)
    ax.set_xlim(freqs[0] / 1000, freqs[-1] / 1000)
    ax.legend(loc="best")

    return lines


def freq_real_imag(ax_re, ax_im, result, *, points=None, quantity="pressure",
                   norm="acc", freq_range=None, label_fmt="z={z:.2f} m",
                   **kwargs):
    """Plot real and imaginary parts on two axes.

    Parameters
    ----------
    ax_re, ax_im : matplotlib Axes
    result : BEMResult
    points, quantity, norm, freq_range, label_fmt, **kwargs
        See ``freq_magnitude``.

    Returns
    -------
    (list of Line2D, list of Line2D)
        Lines on real and imaginary axes.
    """
    indices = _resolve_points(result, points)
    colors = point_colors(len(indices))
    freqs = result.frequencies

    if freq_range is not None:
        fsl = result.freq_slice(*freq_range)
        freqs = freqs[fsl]
    else:
        fsl = slice(None)

    data = result.data(quantity, norm)

    lines_re, lines_im = [], []
    for ci, idx in enumerate(indices):
        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = make_point_label(label_fmt, idx, result.eval_points[idx])

        ln_re = ax_re.plot(freqs / 1000, data[idx, fsl].real, label=label, **kw)
        ln_im = ax_im.plot(freqs / 1000, data[idx, fsl].imag, label=label, **kw)
        lines_re.extend(ln_re)
        lines_im.extend(ln_im)

    for ax, part in [(ax_re, "Real"), (ax_im, "Imag")]:
        format_freq_axis(ax)
        ax.set_ylabel(f"{part} Pressure (Pa/m/s\u00b2)")
        ax.set_xlim(freqs[0] / 1000, freqs[-1] / 1000)
        ax.legend(loc="best")

    return lines_re, lines_im
