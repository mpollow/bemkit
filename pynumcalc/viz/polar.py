"""Directivity polar plots."""

import numpy as np

from pynumcalc.viz._style import freq_colors, CMAP_MAGNITUDE


def polar_directivity(ax, result, freq, *, quantity="pressure", norm="acc",
                      db=True, ref="max", angle_key="theta",
                      label_fmt="{f:.0f} Hz", mirror=False, **kwargs):
    """Plot directivity pattern on a polar axis.

    Parameters
    ----------
    ax : matplotlib PolarAxes
        Must be created with ``subplot_kw={"projection": "polar"}``.
    result : BEMResult
    freq : float or list of float
        Frequency/frequencies in Hz to plot.
    quantity : "pressure" or "velocity"
    norm : "acc" or "raw"
    db : bool
        Plot in dB.
    ref : float or "max"
        dB reference. "max" normalizes each curve to its peak.
    angle_key : "theta", "phi", or "auto"
        Which angle to use as the polar axis. "auto" detects planar data
        and uses the in-plane angle for full 360° plots.
    label_fmt : str
        Format string with ``{f}`` for frequency.
    mirror : bool
        If True, mirror the data across π to create a full 360° pattern.
        Useful when data only spans 0-180° but symmetry is expected.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    list of Line2D
    """
    freqs = np.atleast_1d(freq)
    colors = freq_colors(freqs)

    if angle_key == "theta":
        angles = result.theta
    elif angle_key == "phi":
        angles = result.phi
    elif angle_key == "auto":
        # Auto-detect best angle for planar data
        x = result.eval_points[:, 0]
        y = result.eval_points[:, 1]
        z = result.eval_points[:, 2]

        # Check which plane the data lies in
        y_range = y.max() - y.min()
        x_range = x.max() - x.min()
        z_range = z.max() - z.min()

        # Data in XZ plane (y~0): use angle in XZ plane
        if y_range < 0.01 * max(x_range, z_range):
            angles = np.arctan2(x, z)
        # Data in XY plane (z~0): use angle in XY plane
        elif z_range < 0.01 * max(x_range, y_range):
            angles = np.arctan2(y, x)
        # Data in YZ plane (x~0): use angle in YZ plane
        elif x_range < 0.01 * max(y_range, z_range):
            angles = np.arctan2(z, y)
        else:
            # Not planar, fall back to theta
            angles = result.theta
    else:
        raise ValueError(f"angle_key must be 'theta', 'phi', or 'auto', got {angle_key!r}")

    valid = result.valid_mask
    mag_all = result.magnitude(quantity, norm)

    # Sort by angle for clean line plots
    order = np.argsort(angles[valid])
    ang_sorted = angles[valid][order]

    lines = []
    for ci, f in enumerate(freqs):
        fi = result.at_freq(f)
        mag = mag_all[valid, fi][order]

        if db:
            ref_val = mag.max() if ref == "max" else float(ref)
            with np.errstate(divide="ignore"):
                vals = 20 * np.log10(mag / ref_val)
        else:
            vals = mag

        # Mirror data to create full 360° plot
        if mirror:
            # Create mirrored angles and values
            # Map angle θ to 2π - θ (reflection across vertical axis)
            ang_mirrored = 2 * np.pi - ang_sorted[::-1]
            vals_mirrored = vals[::-1]

            # Concatenate: original [0, π] + mirrored (π, 2π]
            # Skip first point of mirrored to avoid duplicate at π
            ang_plot = np.concatenate([ang_sorted, ang_mirrored[1:]])
            vals_plot = np.concatenate([vals, vals_mirrored[1:]])
        else:
            ang_plot = ang_sorted
            vals_plot = vals

        # Close the loop: append first point at 2π to connect the circle
        # Check if we need to close (span close to 360°)
        angle_span = ang_plot[-1] - ang_plot[0]
        if angle_span > 1.5 * np.pi:  # More than 270°, likely a full circle
            ang_plot = np.append(ang_plot, ang_plot[0] + 2 * np.pi)
            vals_plot = np.append(vals_plot, vals_plot[0])

        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = label_fmt.format(f=f)
        ln = ax.plot(ang_plot, vals_plot, label=label, **kw)
        lines.extend(ln)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    return lines


def polar_balloon_slice(ax, result, freq, *, slice_phi=0.0,
                        quantity="pressure", norm="acc", db=True, ref="max",
                        phi_tol=None, label_fmt="{f:.0f} Hz", **kwargs):
    """Extract a great-circle cut at a given phi from a spherical grid.

    Parameters
    ----------
    ax : matplotlib PolarAxes
    result : BEMResult
    freq : float or list of float
        Frequency/frequencies in Hz.
    slice_phi : float
        Azimuthal angle in degrees for the slice.
    quantity, norm, db, ref, label_fmt, **kwargs
        See ``polar_directivity``.
    phi_tol : float or None
        Tolerance in degrees for phi matching. None = auto from grid spacing.

    Returns
    -------
    list of Line2D
    """
    freqs = np.atleast_1d(freq)
    colors = freq_colors(freqs)

    phi_deg = result.phi_deg
    theta = result.theta

    # Auto tolerance from grid spacing
    if phi_tol is None:
        valid_phi = phi_deg[result.valid_mask]
        unique_phi = np.unique(np.round(valid_phi, 4))
        if len(unique_phi) > 1:
            phi_tol = np.min(np.diff(unique_phi)) / 2
        else:
            phi_tol = 5.0

    # Select points near the requested phi
    mask = result.valid_mask & (np.abs(phi_deg - slice_phi) < phi_tol)

    if mask.sum() == 0:
        return []

    angles = theta[mask]
    order = np.argsort(angles)
    ang_sorted = angles[order]

    mag_all = result.magnitude(quantity, norm)

    lines = []
    for ci, f in enumerate(freqs):
        fi = result.at_freq(f)
        mag = mag_all[mask, fi][order]

        if db:
            ref_val = mag.max() if ref == "max" else float(ref)
            with np.errstate(divide="ignore"):
                vals = 20 * np.log10(mag / ref_val)
        else:
            vals = mag

        kw = {"color": colors[ci], "alpha": 0.8}
        kw.update(kwargs)
        label = label_fmt.format(f=f)
        ln = ax.plot(ang_sorted, vals, label=label, **kw)
        lines.extend(ln)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    return lines
