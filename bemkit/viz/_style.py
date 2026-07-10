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


def point_colors(n, cmap=CMAP_MAGNITUDE):
    """Generate n RGBA colors for line cycling.

    Parameters
    ----------
    n : int
        Number of colors.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    (n, 4) RGBA array
    """
    cm = plt.colormaps[cmap]
    return cm(np.linspace(0, 0.9, max(n, 1)))


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


def format_freq_axis(ax, unit="kHz"):
    """Apply frequency axis formatting.

    Parameters
    ----------
    ax : matplotlib Axes
    unit : "kHz" or "Hz"
    """
    if unit == "kHz":
        ax.set_xlabel(FREQ_LABEL)
    else:
        ax.set_xlabel(FREQ_LABEL_HZ)
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
