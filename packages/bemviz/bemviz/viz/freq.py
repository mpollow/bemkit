"""Frequency-domain plotting functions."""

import numpy as np

from bemviz.viz._angles import DEFAULT_ANGLES, angle_indices, axis_positions
from bemviz.viz._style import (
    point_colors, angle_styles, make_point_label, format_freq_axis,
    PRESSURE_LABEL_DB_ACC, PRESSURE_LABEL, PHASE_LABEL_DEG, PHASE_LABEL_RAD,
    pressure_label,
    ONAXIS_LABEL, ONAXIS_LABEL_DB,
)


def _sparse_kw(result):
    """Marker kwargs when the band has gaps, empty otherwise.

    A partial-band run leaves isolated computed bins with no neighbour to
    draw a line segment to, so without a marker they render as nothing at
    all — the data looks absent rather than sparse.
    """
    return {"marker": "o", "markersize": 4} if result.missing_bins else {}


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


def freq_magnitude(ax, result, *, points=None, quantity="pressure", norm="auto",
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
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
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
        kw.update(_sparse_kw(result))
        kw.update(kwargs)

        label = make_point_label(label_fmt, idx, result.eval_points[idx])

        if db:
            ln = ax.plot(freqs / 1000, vals, label=label, **kw)
        else:
            ln = ax.semilogy(freqs / 1000, vals, label=label, **kw)
        lines.extend(ln)

    format_freq_axis(ax)
    ax.set_ylabel(pressure_label(norm, db))
    ax.set_xlim(freqs[0] / 1000, freqs[-1] / 1000)
    ax.legend(loc="best")

    return lines


def freq_phase(ax, result, *, points=None, quantity="pressure", norm="auto",
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
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
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
        kw.update(_sparse_kw(result))
        kw.update(kwargs)
        label = make_point_label(label_fmt, idx, result.eval_points[idx])
        ln = ax.plot(freqs / 1000, vals, label=label, **kw)
        lines.extend(ln)

    format_freq_axis(ax)
    ax.set_ylabel(PHASE_LABEL_DEG if degrees else PHASE_LABEL_RAD)
    ax.set_xlim(freqs[0] / 1000, freqs[-1] / 1000)
    ax.legend(loc="best")

    return lines


def angle_response(ax, result, *, angles=DEFAULT_ANGLES, quantity="pressure",
                   norm="auto", db=True, ref=1.0, freq_range=None, tol=None,
                   log_freq=True, **kwargs):
    """Plot the frequency response of selected directions of a circular set.

    The default fan — 0°, ±30°, ±60°, ±90° — is the standard off-axis family
    quoted for a loudspeaker.  Requested angles the set does not cover are
    dropped silently, so a half-circle set simply yields fewer curves.

    A ±θ pair shares a color and is told apart by line style; see
    ``_style.angle_styles``.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    angles : sequence of float
        Requested directions in degrees.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    db : bool
        Plot in dB (the default) or on a log magnitude axis.
    ref : "onaxis", "max", "globalmax", or float
        dB reference — see ``directivity.directivity_levels``.  The default
        1.0 keeps the curves absolute, so the on-axis response is visible as
        well as the off-axis spread; ``"onaxis"`` turns the plot into the line
        equivalent of the isobaric map.
    freq_range : (f_min, f_max) or None
        Frequency bounds in Hz.
    tol : float or None
        Angle-matching tolerance in degrees (None = one angular step).
    log_freq : bool
        Logarithmic frequency axis.
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
    from bemviz.viz.directivity import directivity_levels, level_label

    indices, matched = angle_indices(result, angles, tol=tol)
    if len(indices) == 0:
        raise ValueError(f"none of the angles {list(angles)} are covered by "
                         f"fieldpoint set {result.name!r}")

    freqs = result.frequencies
    fsl = slice(None) if freq_range is None else result.freq_slice(*freq_range)
    freqs = freqs[fsl]

    mag = result.magnitude(quantity, norm)[indices][:, fsl]
    vals = directivity_levels(mag, matched, ref) if db else mag

    styles = angle_styles(matched)

    lines = []
    for ci, ang in enumerate(matched):
        kw = {"alpha": 0.9}
        kw.update(styles[ci])
        kw.update(_sparse_kw(result))
        kw.update(kwargs)
        plot = ax.plot if db else ax.semilogy
        lines.extend(plot(freqs, vals[ci], label=f"{ang:+.0f}°", **kw))

    ax.set_xlim(freqs[0], freqs[-1])
    format_freq_axis(ax, unit="Hz", log=log_freq)
    ax.set_ylabel(level_label(ref) if db else pressure_label(norm, db))
    ax.legend(loc="best", ncols=2 if len(matched) > 4 else 1)

    return lines


def onaxis_response(ax, result, *, quantity="pressure", norm="auto", db=True,
                    ref=1.0, freq_range=None, log_freq=True, **kwargs):
    """Plot the distance-compensated response of an on-axis fieldpoint set.

    Every curve is multiplied by its own distance r from the origin.  Pressure
    from a compact source falls as 1/r, so |p|·r is distance-independent: in
    the far field the curves collapse onto one, and where they do *not* — near
    field, or a source large compared with r — the spread is the reading of
    interest.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    db : bool
        Plot in dB (the default) or on a log magnitude axis.
    ref : float
        dB reference in Pa·m per the data's own normalization.
    freq_range : (f_min, f_max) or None
        Frequency bounds in Hz.
    log_freq : bool
        Logarithmic frequency axis.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    list of Line2D

    Raises
    ------
    bemviz.viz._angles.NotOnAxis
        If the fieldpoint set is not collinear through the origin.
    """
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
    from bemviz.viz._angles import AXES

    positions, axis = axis_positions(result)
    # The origin itself carries no distance to compensate with (and normally
    # sits inside the mesh, so it was dropped at export anyway).
    indices = np.flatnonzero(~np.isnan(positions) & (np.abs(positions) > 0))
    order = indices[np.argsort(np.abs(positions[indices]))]

    freqs = result.frequencies
    fsl = slice(None) if freq_range is None else result.freq_slice(*freq_range)
    freqs = freqs[fsl]

    r = np.abs(positions[order])[:, None]
    vals = result.magnitude(quantity, norm)[order][:, fsl] * r
    if db:
        with np.errstate(divide="ignore"):
            vals = 20 * np.log10(vals / float(ref))

    colors = point_colors(len(order), stop=0.75)

    lines = []
    for ci, idx in enumerate(order):
        kw = {"color": colors[ci], "alpha": 0.9}
        kw.update(_sparse_kw(result))
        kw.update(kwargs)
        label = f"{AXES[axis]}={positions[idx]:+.2f} m"
        plot = ax.plot if db else ax.semilogy
        lines.extend(plot(freqs, vals[ci], label=label, **kw))

    ax.set_xlim(freqs[0], freqs[-1])
    format_freq_axis(ax, unit="Hz", log=log_freq)
    ax.set_ylabel(ONAXIS_LABEL_DB if db else ONAXIS_LABEL)
    ax.legend(loc="best")

    return lines


def freq_real_imag(ax_re, ax_im, result, *, points=None, quantity="pressure",
                   norm="auto", freq_range=None, label_fmt="z={z:.2f} m",
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
    # Resolve 'auto' before anything reads or labels the data, so a
    # scattering model plots raw and says so.
    norm = result.effective_norm(norm)
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
