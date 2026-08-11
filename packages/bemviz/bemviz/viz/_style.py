"""Shared style helpers for bemviz.

Reduced to the colormap picker when the 2D plotting family was removed
(2026-08-12). The axis labels, colour cycles and axis formatters that lived
here served `bem MODEL plot`; recover them from git history if it returns.

Deliberately free of matplotlib at import time — these are names, and the 3D
path imports pyplot lazily only where it needs a colormap object.
"""

# -- Colormaps --
CMAP_MAGNITUDE = "viridis"
CMAP_PHASE = "hsv"
CMAP_REAL_IMAG = "RdBu_r"
CMAP_FIELD = "magma"


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
