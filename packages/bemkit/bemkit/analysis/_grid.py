"""Grid detection and reshape utilities for structured evaluation grids."""

import numpy as np


def detect_grid(points):
    """Detect if points form a regular grid (Cartesian or spherical).

    Parameters
    ----------
    points : (N, 3) array
        Evaluation point coordinates. May contain NaN rows.

    Returns
    -------
    dict or None
        Grid info dict with keys: type, shape, axes, row_values, col_values.
        None if no grid structure detected.
    """
    valid = ~np.any(np.isnan(points), axis=1)
    pts = points[valid]
    n_total = len(points)
    n_valid = len(pts)

    if n_valid < 4:
        return None

    # Try Cartesian: two of (x,y,z) vary, one constant
    cart = _detect_cartesian(pts, n_total, n_valid)
    if cart is not None:
        return cart

    # Try spherical: constant radius, theta × phi grid
    sph = _detect_spherical(pts, n_total, n_valid)
    if sph is not None:
        return sph

    return None


def _detect_cartesian(pts, n_total, n_valid):
    """Detect Cartesian grid where one axis is constant."""
    labels = ("x", "y", "z")

    for const_ax in range(3):
        vals = pts[:, const_ax]
        if np.ptp(vals) > 1e-10 * max(np.max(np.abs(vals)), 1e-12):
            continue

        # Two varying axes
        vary = [i for i in range(3) if i != const_ax]
        a_vals = np.unique(np.round(pts[:, vary[0]], 10))
        b_vals = np.unique(np.round(pts[:, vary[1]], 10))
        na, nb = len(a_vals), len(b_vals)

        # Check: valid points fit na×nb (no NaN rows) or n_total is a
        # perfect square / product that contains these unique values
        if na * nb == n_valid == n_total:
            # All points present, simple case
            pass
        elif na * nb == n_valid and n_valid < n_total:
            # NaN rows exist but valid points span na × nb — however the
            # original grid has n_total points. Infer full grid dimensions.
            full_na, full_nb = _infer_full_grid(n_total, a_vals, b_vals)
            if full_na is None:
                continue
            a_vals = np.linspace(a_vals.min(), a_vals.max(), full_na)
            b_vals = np.linspace(b_vals.min(), b_vals.max(), full_nb)
            na, nb = full_na, full_nb
        elif n_total > n_valid:
            # Try to factor n_total directly
            full_na, full_nb = _infer_full_grid(n_total, a_vals, b_vals)
            if full_na is None:
                continue
            a_vals = np.linspace(a_vals.min(), a_vals.max(), full_na)
            b_vals = np.linspace(b_vals.min(), b_vals.max(), full_nb)
            na, nb = full_na, full_nb
        else:
            continue

        return {
            "type": "cartesian",
            "shape": (nb, na),  # (rows, cols) = (second_vary, first_vary)
            "axes": (labels[vary[0]], labels[vary[1]]),
            "const_axis": labels[const_ax],
            "const_value": float(pts[0, const_ax]),
            "row_values": b_vals,
            "col_values": a_vals,
        }

    return None


def _infer_full_grid(n_total, a_vals, b_vals):
    """Try to find (na, nb) such that na * nb == n_total.

    Uses the observed range and spacing hints from valid unique values.
    """
    # Check if n_total is a perfect square first
    sqrt_n = int(round(np.sqrt(n_total)))
    if sqrt_n * sqrt_n == n_total:
        return sqrt_n, sqrt_n

    # Try to infer from spacing: if valid values have regular spacing,
    # extrapolate to full range
    for a_count in range(2, n_total):
        if n_total % a_count != 0:
            continue
        b_count = n_total // a_count
        if b_count < 2:
            break
        # Check if this factorization is plausible given unique counts
        if a_count >= len(a_vals) and b_count >= len(b_vals):
            return a_count, b_count

    return None, None


def _detect_spherical(pts, n_total, n_valid):
    """Detect spherical grid with constant radius and theta × phi structure."""
    r = np.sqrt(np.sum(pts**2, axis=1))
    if np.ptp(r) > 1e-6 * np.mean(r):
        return None

    theta = np.arccos(np.clip(pts[:, 2] / r, -1, 1))
    phi = np.arctan2(pts[:, 1], pts[:, 0])

    theta_vals = np.unique(np.round(theta, 8))
    phi_vals = np.unique(np.round(phi, 8))
    nt, nphi = len(theta_vals), len(phi_vals)

    # The key check: n_theta × n_phi must equal total point count
    if nt * nphi != n_total:
        return None

    return {
        "type": "spherical",
        "shape": (nt, nphi),  # (n_theta, n_phi)
        "axes": ("theta", "phi"),
        "radius": float(np.mean(r)),
        "row_values": theta_vals,
        "col_values": phi_vals,
    }


def reshape_to_grid(points, values, grid_info):
    """Reshape flat arrays into 2D grid arrays for plotting.

    Parameters
    ----------
    points : (N, 3) array
        Evaluation points (may contain NaN rows).
    values : (N,) or (N, ...) array
        Values to reshape. NaN preserved for dropped points.
    grid_info : dict
        Output from detect_grid().

    Returns
    -------
    row2d, col2d, val2d : 2D arrays
        Meshgrid coordinates and reshaped values.
    """
    nr, nc = grid_info["shape"]

    if grid_info["type"] == "cartesian":
        col2d, row2d = np.meshgrid(grid_info["col_values"],
                                    grid_info["row_values"])
        ax0, ax1 = grid_info["axes"]
        ax_map = {"x": 0, "y": 1, "z": 2}
        ci, ri = ax_map[ax0], ax_map[ax1]

        val2d = _reorder_to_grid(points, values, nr, nc,
                                 ri, grid_info["row_values"],
                                 ci, grid_info["col_values"])

    elif grid_info["type"] == "spherical":
        col2d, row2d = np.meshgrid(grid_info["col_values"],
                                    grid_info["row_values"])
        r = np.sqrt(np.sum(points**2, axis=1))
        theta = np.arccos(np.clip(points[:, 2] / np.where(r > 0, r, 1), -1, 1))
        phi = np.arctan2(points[:, 1], points[:, 0])

        val2d = _reorder_to_grid_spherical(
            theta, phi, values, nr, nc,
            grid_info["row_values"], grid_info["col_values"])

    else:
        raise ValueError(f"Unknown grid type: {grid_info['type']}")

    return row2d, col2d, val2d


def _reorder_to_grid(points, values, nr, nc, row_ax, row_vals, col_ax, col_vals):
    """Map flat points to 2D grid positions by coordinate matching."""
    shape = (nr, nc) + values.shape[1:] if values.ndim > 1 else (nr, nc)
    out = np.full(shape, np.nan)

    for i in range(len(points)):
        if np.any(np.isnan(points[i])):
            continue

        ci = np.argmin(np.abs(col_vals - np.round(points[i, col_ax], 10)))
        ri = np.argmin(np.abs(row_vals - np.round(points[i, row_ax], 10)))
        out[ri, ci] = values[i]

    return out


def _reorder_to_grid_spherical(theta, phi, values, nr, nc, theta_vals, phi_vals):
    """Map flat spherical points to (theta, phi) grid positions."""
    shape = (nr, nc) + values.shape[1:] if values.ndim > 1 else (nr, nc)
    out = np.full(shape, np.nan)

    for i in range(len(theta)):
        if np.isnan(theta[i]):
            continue

        ri = np.argmin(np.abs(theta_vals - np.round(theta[i], 8)))
        ci = np.argmin(np.abs(phi_vals - np.round(phi[i], 8)))
        out[ri, ci] = values[i]

    return out
