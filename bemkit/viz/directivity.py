"""Directivity-over-frequency plots for circular fieldpoint sets.

The isobaric plot is the one map that shows a source's whole directivity
behaviour at once: angle on the vertical axis, frequency on the horizontal,
level in color.  Beamwidth narrowing with frequency, off-axis nulls and lobing
all read directly off it, where a stack of polar plots only samples them.
"""

import numpy as np

from bemkit.viz._angles import circle_angles
from bemkit.viz._style import (
    ANGLE_LABEL, CMAP_MAGNITUDE, LEVEL_LABEL_ABS, LEVEL_LABEL_MAX,
    LEVEL_LABEL_ONAXIS, format_freq_axis,
)


_LEVEL_LABELS = {
    "onaxis": LEVEL_LABEL_ONAXIS,
    "max": LEVEL_LABEL_MAX,
    "globalmax": LEVEL_LABEL_MAX,
}


def level_label(ref="onaxis"):
    """Axis/colorbar label describing a ``ref`` choice."""
    return _LEVEL_LABELS.get(ref, LEVEL_LABEL_ABS)


def directivity_levels(mag, angles_deg, ref="onaxis"):
    """Convert per-angle magnitudes to dB against a directivity reference.

    Parameters
    ----------
    mag : (na, nf) float array
        Magnitudes, rows ordered by ``angles_deg``.
    angles_deg : (na,) float array
        Angle of each row, in degrees.
    ref : "onaxis", "max", "globalmax", or float
        - ``"onaxis"``  — each frequency referenced to its own 0° value, so
          the plot shows directivity only and the on-axis response is flat at
          0 dB.  This is the conventional normalization.
        - ``"max"``     — each frequency referenced to its own peak over angle.
          Same beamwidth reading for a source that peaks on axis, but honest
          about one that does not.
        - ``"globalmax"`` — one reference for the whole map, so the frequency
          response stays visible in the colors alongside the directivity.
        - float         — absolute reference in the data's own units.

    Returns
    -------
    levels : (na, nf) float array
    """
    mag = np.asarray(mag, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)

    if ref == "onaxis":
        on = int(np.argmin(np.abs((angles_deg + 180.0) % 360.0 - 180.0)))
        ref_vals = mag[on][None, :]
    elif ref == "max":
        ref_vals = np.nanmax(mag, axis=0, keepdims=True)
    elif ref == "globalmax":
        ref_vals = np.nanmax(mag)
    else:
        ref_vals = float(ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(mag / ref_vals)


def _edges(centers):
    """Cell edges from cell centers, for ``pcolormesh`` with flat shading."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mid = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[centers[0] - (mid[0] - centers[0])],
                           mid,
                           [centers[-1] + (centers[-1] - mid[-1])]])


def isobaric(ax, result, *, quantity="pressure", norm="acc", ref="onaxis",
             freq_range=None, clim=(-30.0, 0.0), cmap=CMAP_MAGNITUDE,
             contours=None, contour_kwargs=None, log_freq=True, **kwargs):
    """Plot level over angle and frequency as a color map ("sonogram").

    The set must be circular about the origin — any angular resolution, any of
    the three coordinate planes.  Angles come from
    ``bemkit.viz._angles.circle_angles``, so a ``dims = "zx"`` polar set has 0°
    on +z.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    ref : "onaxis", "max", "globalmax", or float
        dB reference — see ``directivity_levels``.
    freq_range : (f_min, f_max) or None
        Frequency bounds in Hz.
    clim : (lo, hi) or None
        Color limits in dB.  None auto-scales to the data.
    cmap : str
        Sequential colormap — level is a magnitude, so it takes a single-hue
        perceptually uniform ramp.
    contours : float, sequence of float, or None
        Iso-level contour lines: a float is a step in dB, a sequence is the
        levels themselves.  None draws none.
    contour_kwargs : dict or None
        Overrides for the contour lines.
    log_freq : bool
        Logarithmic frequency axis (octaves get equal width).
    **kwargs
        Passed to ``ax.pcolormesh``.

    Returns
    -------
    mesh : QuadMesh
        Pass to ``fig.colorbar``; ``level_label(ref)`` gives its label.

    Raises
    ------
    bemkit.viz._angles.NotCircular
        If the fieldpoint set is not a circle about the origin.
    """
    angles, _radius, _normal = circle_angles(result)

    valid = ~np.isnan(angles)
    order = np.argsort(angles[valid])
    ang = angles[valid][order]

    freqs = result.frequencies
    fsl = slice(None) if freq_range is None else result.freq_slice(*freq_range)
    freqs = freqs[fsl]

    mag = result.magnitude(quantity, norm)[valid][:, fsl][order]
    levels = directivity_levels(mag, ang, ref)

    vmin, vmax = (None, None) if clim is None else clim
    kw = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "shading": "flat",
          "rasterized": True}
    kw.update(kwargs)
    mesh = ax.pcolormesh(_edges(freqs), _edges(ang), levels, **kw)

    if contours is not None:
        levels_at = _contour_levels(contours, levels, clim)
        if len(levels_at):
            # linestyles="solid": matplotlib dashes negative levels by
            # default, and every level here is negative.
            ckw = {"colors": "k", "linewidths": 0.6, "alpha": 0.45,
                   "linestyles": "solid"}
            ckw.update(contour_kwargs or {})
            ax.contour(freqs, ang, levels, levels=levels_at, **ckw)

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(ang[0], ang[-1])
    format_freq_axis(ax, unit="Hz", log=log_freq, grid=False)
    ax.set_ylabel(ANGLE_LABEL)
    ax.set_yticks(_angle_ticks(ang))

    return mesh


def _angle_ticks(angles, step=30.0):
    """Ticks every ``step`` degrees inside the covered arc."""
    lo = np.ceil(angles[0] / step) * step
    hi = np.floor(angles[-1] / step) * step
    ticks = np.arange(lo, hi + step / 2, step)
    return ticks if ticks.size >= 3 else np.asarray([angles[0], 0.0, angles[-1]])


def _contour_levels(contours, levels, clim):
    """Resolve the ``contours`` argument into explicit dB levels."""
    if np.ndim(contours) > 0:
        return np.asarray(contours, dtype=float)

    step = float(contours)
    lo, hi = clim if clim is not None else (np.nanmin(levels), np.nanmax(levels))
    start = np.ceil(lo / step) * step
    out = np.arange(start, hi + step / 2, step)
    # A line exactly on the top of the scale traces the reference itself and
    # only adds ink.
    return out[out < hi - step / 4]
