"""2D spatial field plots: heatmap, contour, scatter."""

import numpy as np

from pynumcalc.viz._style import CMAP_FIELD, CMAP_PHASE, CMAP_MAGNITUDE, cmap_for_quantity


def _get_values(result, freq, value_func, quantity, norm):
    """Extract (N,) values at a frequency using a value function.

    Parameters
    ----------
    result : BEMResult
    freq : float
        Frequency in Hz.
    value_func : str or callable
        "magnitude", "magnitude_db", "phase", "real", "imag", or (result, fi) -> (N,).
    quantity, norm : str
        Passed to BEMResult data accessors.

    Returns
    -------
    (N,) float array
    """
    fi = result.at_freq(freq)

    if callable(value_func):
        return value_func(result, fi)

    funcs = {
        "magnitude": lambda: result.magnitude(quantity, norm)[:, fi],
        "magnitude_db": lambda: result.magnitude_db(quantity, norm)[:, fi],
        "phase": lambda: np.degrees(result.phase(quantity, norm)[:, fi]),
        "real": lambda: result.data(quantity, norm)[:, fi].real,
        "imag": lambda: result.data(quantity, norm)[:, fi].imag,
    }
    if value_func not in funcs:
        raise ValueError(f"Unknown value_func: {value_func!r}. "
                         f"Use {list(funcs.keys())} or a callable.")
    return funcs[value_func]()


def _default_cmap(value_func):
    """Pick a sensible default colormap for the given value function."""
    if isinstance(value_func, str):
        return cmap_for_quantity(value_func)
    return CMAP_FIELD


def heatmap(ax, result, freq, *, quantity="pressure", norm="acc",
            value_func="magnitude_db", cmap=None, vmin=None, vmax=None,
            colorbar=True, **kwargs):
    """Plot a 2D heatmap on a structured grid.

    Requires ``result.is_grid``. NaN regions are transparent.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    freq : float
        Frequency in Hz.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    value_func : str or callable
        See ``_get_values``.
    cmap : str or None
        Colormap. None = auto from value_func.
    vmin, vmax : float or None
        Color limits.
    colorbar : bool
        Add colorbar.
    **kwargs
        Passed to ``ax.pcolormesh()``.

    Returns
    -------
    QuadMesh
    """
    vals = _get_values(result, freq, value_func, quantity, norm)
    row2d, col2d, val2d = result.as_grid(vals)

    if cmap is None:
        cmap = _default_cmap(value_func)

    grid_info = result._grid_info
    kw = {"cmap": cmap, "shading": "auto"}
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    kw.update(kwargs)

    mesh = ax.pcolormesh(col2d, row2d, val2d, **kw)

    if colorbar:
        ax.figure.colorbar(mesh, ax=ax)

    # Label axes from grid info
    if grid_info["type"] == "cartesian":
        ax.set_xlabel(f"{grid_info['axes'][0]} (m)")
        ax.set_ylabel(f"{grid_info['axes'][1]} (m)")
    elif grid_info["type"] == "spherical":
        ax.set_xlabel(f"{grid_info['axes'][1]} (rad)")
        ax.set_ylabel(f"{grid_info['axes'][0]} (rad)")

    ax.set_aspect("equal")
    return mesh


def contour(ax, result, freq, *, quantity="pressure", norm="acc",
            value_func="magnitude_db", levels=20, filled=True,
            cmap=None, colorbar=True, **kwargs):
    """Plot contour lines or filled contours on a structured grid.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    freq : float
    quantity, norm, value_func, cmap, colorbar
        See ``heatmap``.
    levels : int or array-like
        Number of contour levels or explicit values.
    filled : bool
        Use ``contourf`` (True) or ``contour`` (False).
    **kwargs
        Passed to contour/contourf.

    Returns
    -------
    QuadContourSet
    """
    vals = _get_values(result, freq, value_func, quantity, norm)
    row2d, col2d, val2d = result.as_grid(vals)

    if cmap is None:
        cmap = _default_cmap(value_func)

    kw = {"cmap": cmap, "levels": levels}
    kw.update(kwargs)

    grid_info = result._grid_info
    if filled:
        cs = ax.contourf(col2d, row2d, val2d, **kw)
    else:
        cs = ax.contour(col2d, row2d, val2d, **kw)

    if colorbar:
        ax.figure.colorbar(cs, ax=ax)

    if grid_info["type"] == "cartesian":
        ax.set_xlabel(f"{grid_info['axes'][0]} (m)")
        ax.set_ylabel(f"{grid_info['axes'][1]} (m)")

    ax.set_aspect("equal")
    return cs


def scatter_field(ax, result, freq, *, quantity="pressure", norm="acc",
                  value_func="magnitude_db", coords=("x", "z"),
                  cmap=None, size=20, colorbar=True, **kwargs):
    """Scatter plot for arbitrary point distributions.

    Parameters
    ----------
    ax : matplotlib Axes
    result : BEMResult
    freq : float
    quantity, norm, value_func, cmap, colorbar
        See ``heatmap``.
    coords : (str, str)
        Pair of coordinate names for the 2D projection.
        Valid: "x", "y", "z", "r", "theta", "phi".
    size : float
        Marker size.
    **kwargs
        Passed to ``ax.scatter()``.

    Returns
    -------
    PathCollection
    """
    coord_map = {
        "x": result.x, "y": result.y, "z": result.z,
        "r": result.r, "theta": result.theta_deg, "phi": result.phi_deg,
    }

    cx, cy = coords
    xv, yv = coord_map[cx], coord_map[cy]

    vals = _get_values(result, freq, value_func, quantity, norm)
    valid = result.valid_mask

    if cmap is None:
        cmap = _default_cmap(value_func)

    kw = {"c": vals[valid], "cmap": cmap, "s": size}
    kw.update(kwargs)

    sc = ax.scatter(xv[valid], yv[valid], **kw)

    if colorbar:
        ax.figure.colorbar(sc, ax=ax)

    ax.set_xlabel(cx)
    ax.set_ylabel(cy)
    return sc


def spherical_heatmap(ax, result, freq, *, quantity="pressure", norm="acc",
                      value_func="magnitude_db", cmap=None, vmin=None, vmax=None,
                      colorbar=True, projection=None, **kwargs):
    """Theta-phi heatmap for spherical grids.

    Parameters
    ----------
    ax : matplotlib Axes
        For Mollweide/Hammer, create with appropriate projection.
    result : BEMResult
    freq : float
    quantity, norm, value_func, cmap, vmin, vmax, colorbar
        See ``heatmap``.
    projection : None
        Reserved for future projection support.
    **kwargs
        Passed to ``ax.pcolormesh()``.

    Returns
    -------
    QuadMesh
    """
    vals = _get_values(result, freq, value_func, quantity, norm)
    row2d, col2d, val2d = result.as_grid(vals)
    # row2d = theta, col2d = phi (from spherical grid detection)

    if cmap is None:
        cmap = _default_cmap(value_func)

    kw = {"cmap": cmap, "shading": "auto"}
    if vmin is not None:
        kw["vmin"] = vmin
    if vmax is not None:
        kw["vmax"] = vmax
    kw.update(kwargs)

    # Plot phi on x-axis, theta on y-axis (convention: azimuth horizontal)
    mesh = ax.pcolormesh(np.degrees(col2d), np.degrees(row2d), val2d, **kw)

    if colorbar:
        ax.figure.colorbar(mesh, ax=ax)

    ax.set_xlabel("Azimuth \u03c6 (\u00b0)")
    ax.set_ylabel("Elevation \u03b8 (\u00b0)")
    return mesh
