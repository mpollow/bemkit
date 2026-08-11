"""Style constants and helpers for BEM plots."""

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -- Colormaps --
CMAP_MAGNITUDE = "viridis"
CMAP_PHASE = "hsv"
CMAP_REAL_IMAG = "RdBu_r"
CMAP_FIELD = "magma"

# -- Axis labels --
FREQ_LABEL = "Frequency (kHz)"
FREQ_LABEL_HZ = "Frequency (Hz)"
PRESSURE_LABEL = "|Pressure| (Pa)"
PRESSURE_LABEL_DB = "Pressure (dB)"
PRESSURE_LABEL_DB_ACC = "|Pressure| (Pa/m/s\u00b2)"


PHASE_LABEL_DEG = "Phase (\u00b0)"
PHASE_LABEL_RAD = "Phase (rad)"
TIME_LABEL_MS = "Time (ms)"
TIME_LABEL_S = "Time (s)"
IR_LABEL = "Pressure (Pa/m/s\u00b2)"
ANGLE_LABEL = "Angle (\u00b0)"
LEVEL_LABEL_ONAXIS = "Level re on-axis (dB)"
LEVEL_LABEL_MAX = "Level re peak (dB)"
LEVEL_LABEL_ABS = "Level (dB re 1 Pa/m/s\u00b2)"
ONAXIS_LABEL_DB = "20\u00b7log\u2081\u2080(|p|\u00b7r) (dB re 1 Pa\u00b7m/m/s\u00b2)"
ONAXIS_LABEL = "|p|\u00b7r (Pa\u00b7m/m/s\u00b2)"


def pressure_label(norm="acc", db=True):
    """Axis label matching the normalization a plot actually used.

    A scattering model has no acc normalization, so its plots show raw
    Pascals; labelling those "Pa/m/s\u00b2" would misstate the units by a
    frequency-dependent factor. Callers resolve ``"auto"`` through
    ``BEMResult.effective_norm`` first, so only "acc"/"raw" reach here.
    """
    if norm == "raw":
        return PRESSURE_LABEL_DB if db else PRESSURE_LABEL
    return PRESSURE_LABEL_DB_ACC


def ir_label(norm="acc"):
    """Time-domain amplitude label, matching the normalization used."""
    return "Pressure (Pa)" if norm == "raw" else IR_LABEL


def cmap_for_quantity(quantity):
    """Pick a sensible colormap for the given quantity string.

    Parameters
    ----------
    quantity : str
        One of "phase", "real", "imag", "magnitude_db", "magnitude", etc.

    Returns
    -------
    str
        Colormap name.
    """
    if quantity == "phase":
        return CMAP_PHASE
    if quantity in ("real", "imag"):
        return CMAP_REAL_IMAG
    return CMAP_FIELD


def point_colors(n, cmap=CMAP_MAGNITUDE, stop=0.9):
    """Generate n RGBA colors for line cycling.

    Parameters
    ----------
    n : int
        Number of colors.
    cmap : str
        Matplotlib colormap name.
    stop : float
        Where to stop in the colormap. Below ~0.8 the pale end of viridis is
        avoided, which keeps the last line legible on white.

    Returns
    -------
    (n, 4) RGBA array
    """
    cm = plt.colormaps[cmap]
    return cm(np.linspace(0, stop, max(n, 1)))


def freq_colors(freqs, cmap=CMAP_MAGNITUDE):
    """Map frequency values to RGBA colors.

    Parameters
    ----------
    freqs : array-like
        Frequency values.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    (n, 4) RGBA array
    """
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        return np.empty((0, 4))
    norm = mpl.colors.Normalize(vmin=freqs.min(), vmax=freqs.max())
    cm = plt.colormaps[cmap]
    return cm(norm(freqs))


def angle_styles(angles_deg, cmap=CMAP_MAGNITUDE):
    """Line styles for a fan of off-axis angles.

    A ±θ pair shares one color and is separated by line style (solid for the
    positive angle, dashed for the negative), so the pairing survives both a
    grayscale print and every form of color blindness.  Color runs light→dark
    with |θ|, i.e. the on-axis curve is the darkest.

    Parameters
    ----------
    angles_deg : sequence of float
    cmap : str

    Returns
    -------
    list of dict
        ``{"color": rgba, "linestyle": str}`` per angle, ready to splat into
        ``ax.plot``.
    """
    angles = np.asarray(angles_deg, dtype=float)
    if angles.size == 0:
        return []

    # Round before keying: angles come from arctan2 of generated coordinates,
    # so +30° and -30° differ in the last bits and would otherwise be given
    # two different colors — exactly the pairing the plot is meant to show.
    mags = np.unique(np.round(np.abs(angles), 6))
    cm = plt.colormaps[cmap]
    # Darkest at 0°, lightest at the widest angle: the on-axis curve is the
    # reference every other one is read against, so it gets the most
    # contrast. Stop short of the pale end of viridis so the widest angle
    # stays legible on white too.
    levels = np.linspace(0.0, 0.75, len(mags))
    color_of = {m: cm(lv) for m, lv in zip(mags, levels)}

    return [{"color": color_of[round(abs(a), 6)],
             "linestyle": "-" if a >= 0 else "--"}
            for a in angles]


def make_point_label(fmt, idx, point):
    """Format a label string for a point.

    Supported format keys: {i}, {x}, {y}, {z}, {r}, {theta}, {phi}.

    Parameters
    ----------
    fmt : str or callable
        Format string or ``(idx, point_xyz) -> str`` callable.
    idx : int
        Point index.
    point : (3,) array
        Cartesian coordinates.
    """
    if callable(fmt):
        return fmt(idx, point)

    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arccos(np.clip(z / r, -1, 1))) if r > 0 else 0.0
    phi = np.degrees(np.arctan2(y, x))
    return fmt.format(i=idx, x=x, y=y, z=z, r=r, theta=theta, phi=phi)


def format_freq_axis(ax, unit="kHz", *, log=False, grid=True):
    """Apply frequency axis formatting.

    Parameters
    ----------
    ax : matplotlib Axes
    unit : "kHz" or "Hz"
    log : bool
        Logarithmic frequency axis with 1/1-octave ticks labelled in plain
        numbers (63, 125, … 1k, 2k) rather than powers of ten.
    grid : bool
        Draw the recessive frequency grid.  Off for image plots, where the
        grid would sit on top of the data.
    """
    if unit == "kHz":
        ax.set_xlabel(FREQ_LABEL)
    else:
        ax.set_xlabel(FREQ_LABEL_HZ)

    if log:
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ticks = [f for f in _OCTAVE_TICKS if lo <= f <= hi]
        if len(ticks) >= 3:
            ax.set_xticks(ticks)
            ax.set_xticklabels([_freq_tick_label(f, unit) for f in ticks])
        ax.minorticks_off()

    # ax.grid(False, alpha=...) warns and turns the grid *on*; pass line
    # properties only when the grid is actually wanted.
    ax.grid(True, alpha=0.3) if grid else ax.grid(False)


_OCTAVE_TICKS = [15.625, 31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000,
                 16000, 32000]


def _freq_tick_label(f_hz, unit):
    """Plain-number tick label for a log frequency axis."""
    f = f_hz / 1000 if unit == "kHz" else f_hz
    if unit == "Hz" and f_hz >= 1000:
        return f"{f_hz / 1000:g}k"
    return f"{f:g}"


def format_polar_axes(ax, *, zero="top", clockwise=False, rlim=None,
                      tick_step=30):
    """Set up a polar axis for a directivity plot.

    Parameters
    ----------
    ax : matplotlib PolarAxes
    zero : "top", "right", "bottom", or "left"
        Where 0° sits.  "top" is the convention for a source radiating along
        its own axis.
    clockwise : bool
        Direction of increasing angle.
    rlim : (lo, hi) or None
        Radial (level) limits in dB.
    tick_step : float
        Angular tick spacing in degrees.
    """
    offsets = {"right": 0.0, "top": np.pi / 2, "left": np.pi, "bottom": -np.pi / 2}
    if zero not in offsets:
        raise ValueError(f"zero must be one of {list(offsets)}, got {zero!r}")

    ax.set_theta_offset(offsets[zero])
    ax.set_theta_direction(-1 if clockwise else 1)

    ticks = np.arange(-180, 180, tick_step)
    ax.set_thetagrids(ticks, [f"{t:g}°" for t in ticks])
    # A negative tick makes set_thetagrids widen the theta limits to
    # (-180°, 360°) — a 540° sweep that draws the pattern into two thirds of
    # the circle. Restore the full turn *after* setting the grids.
    ax.set_thetalim(-np.pi, np.pi)
    # Keep the dB scale off the 0° spoke, where the main lobe runs.
    ax.set_rlabel_position(135)
    if rlim is not None:
        ax.set_rlim(*rlim)
    ax.grid(True, alpha=0.3)


@contextmanager
def bem_style():
    """Context manager applying BEM plot rc defaults."""
    overrides = {
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "font.size": 11,
    }
    with mpl.rc_context(overrides):
        yield
