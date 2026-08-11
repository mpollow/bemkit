"""Angle geometry for fieldpoint sets.

The plots in this package that are *tailored to a dataset* — isobaric maps,
polar responses, off-axis frequency responses, distance-compensated on-axis
responses — all need the same two questions answered about a set of evaluation
points:

    circular set   — do the points lie on a circle centred on the origin, and
                     what is each point's signed in-plane angle?
    on-axis set    — are the points collinear through the origin, and what is
                     each point's signed distance?

Both are answered from the coordinates alone: a ``BEMResult`` carries no record
of the ``[fieldpoints.<name>]`` table that generated it.

Angle convention (matches ``polar_directivity(angle_key="auto")``): for a
circle whose plane normal is axis *n*, 0° lies on axis ``(n + 1) % 3`` and +90°
on axis ``(n + 2) % 3`` — a right-handed rotation about the normal.  So a
``dims = "zx"`` circle (``make_fields_polar``) has its 0° on +z, which is the
on-axis direction for a source radiating along +z.
"""

import numpy as np

AXES = "xyz"

# Relative tolerances for "flat in one axis" and "constant radius".
_PLANE_TOL = 1e-3
_RADIUS_TOL = 1e-2
# Relative tolerance for "collinear through the origin".
_AXIS_TOL = 1e-6


class NotCircular(ValueError):
    """Raised when a fieldpoint set is not a circle about the origin."""


class NotOnAxis(ValueError):
    """Raised when a fieldpoint set is not collinear through the origin."""


def _valid_points(result):
    """(mask, points[mask]) — drop the NaN rows left by fieldpoint validation."""
    mask = result.valid_mask
    return mask, result.eval_points[mask]


def circle_plane(points, *, tol=_PLANE_TOL):
    """Index of the axis normal to a planar point set through the origin.

    Returns None when the points are not planar, when more than one axis is
    degenerate (a line, not a circle), or when the plane misses the origin.
    """
    if len(points) < 3:
        return None

    extents = np.ptp(points, axis=0)
    scale = extents.max()
    if scale <= 0:
        return None

    flat = np.flatnonzero(extents <= tol * scale)
    if flat.size != 1:
        return None

    normal = int(flat[0])
    # The plane must pass through the origin: angles are measured about it.
    if np.max(np.abs(points[:, normal])) > tol * scale:
        return None
    return normal


def circle_angles(result, *, tol=_PLANE_TOL, radius_tol=_RADIUS_TOL):
    """Signed in-plane angle of every point of a circular fieldpoint set.

    Parameters
    ----------
    result : BEMResult
    tol, radius_tol : float
        Relative tolerances for planarity and for constant radius.

    Returns
    -------
    angles_deg : (N,) float array
        Angle in degrees in [-180, 180), NaN for dropped points.  Aligned with
        ``result.eval_points``, so it indexes the same way as the data arrays.
    radius : float
        Mean radius in meters.
    normal : int
        Index of the axis normal to the circle's plane (0=x, 1=y, 2=z).

    Raises
    ------
    NotCircular
        If the points are not a circle centred on the origin.
    """
    mask, pts = _valid_points(result)
    normal = circle_plane(pts, tol=tol)
    if normal is None:
        raise NotCircular(
            f"fieldpoint set {result.name!r} is not planar about the origin — "
            "an isobaric/polar plot needs a circular set "
            '(e.g. [fieldpoints.polar] type = "polar")')

    r = np.linalg.norm(pts, axis=1)
    radius = float(r.mean())
    if radius <= 0 or r.std() > radius_tol * radius:
        raise NotCircular(
            f"fieldpoint set {result.name!r} lies in the "
            f"{AXES[(normal + 1) % 3]}{AXES[(normal + 2) % 3]} plane but not on "
            f"a circle (radius {r.min():.3f}–{r.max():.3f} m)")

    zero_ax = (normal + 1) % 3
    quad_ax = (normal + 2) % 3

    angles = np.full(len(result.eval_points), np.nan)
    angles[mask] = np.degrees(np.arctan2(pts[:, quad_ax], pts[:, zero_ax]))
    # arctan2 returns 180° for the point exactly behind; keep the range
    # half-open at -180 so the two ends of a full circle are not duplicated.
    angles[angles >= 180.0] -= 360.0
    return angles, radius, normal


def is_circular(result):
    """True if ``circle_angles`` would succeed for this set."""
    try:
        circle_angles(result)
    except NotCircular:
        return False
    return True


def axis_positions(result, *, tol=_AXIS_TOL):
    """Signed positions of an on-axis (collinear, origin-crossing) set.

    Returns
    -------
    positions : (N,) float array
        Signed distance from the origin along the set's axis, in meters.
        NaN for dropped points.
    axis : int
        Index of the axis the points lie on (0=x, 1=y, 2=z).

    Raises
    ------
    NotOnAxis
        If the points are not collinear along a coordinate axis through the
        origin.
    """
    mask, pts = _valid_points(result)
    if len(pts) < 2:
        raise NotOnAxis(f"fieldpoint set {result.name!r} has too few points")

    extents = np.ptp(pts, axis=0)
    scale = extents.max()
    varying = np.flatnonzero(extents > tol * scale)
    if varying.size != 1:
        raise NotOnAxis(
            f"fieldpoint set {result.name!r} is not collinear — a distance-"
            "compensated on-axis response needs points along one axis "
            '(e.g. [fieldpoints.onaxis] type = "onaxis")')

    axis = int(varying[0])
    off = np.delete(pts, axis, axis=1)
    if np.max(np.abs(off)) > tol * scale:
        raise NotOnAxis(
            f"fieldpoint set {result.name!r} runs parallel to {AXES[axis]} but "
            "is offset from the origin, so distance compensation is undefined")

    positions = np.full(len(result.eval_points), np.nan)
    positions[mask] = pts[:, axis]
    return positions, axis


def is_onaxis(result):
    """True if ``axis_positions`` would succeed for this set."""
    try:
        axis_positions(result)
    except NotOnAxis:
        return False
    return True


DEFAULT_ANGLES = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0)


def angle_indices(result, angles_deg=DEFAULT_ANGLES, *, tol=None):
    """Point indices nearest to each requested angle of a circular set.

    Parameters
    ----------
    result : BEMResult
    angles_deg : sequence of float
        Requested angles in degrees.
    tol : float or None
        Maximum accepted mismatch in degrees.  None derives it from the set's
        own angular spacing (one step), so a request that falls between two
        measured directions still snaps, but one that lies outside the covered
        arc is dropped.

    Returns
    -------
    indices : (k,) int array
        Indices into ``result.eval_points``.
    matched_deg : (k,) float array
        The actual angle of each selected point.

    Notes
    -----
    Requests with no point inside ``tol`` are dropped, not raised on: a half-
    circle set legitimately has no -90° direction.
    """
    angles, _, _ = circle_angles(result)
    have = np.flatnonzero(~np.isnan(angles))
    if have.size == 0:
        return np.empty(0, dtype=int), np.empty(0)

    if tol is None:
        spacing = np.diff(np.sort(angles[have]))
        spacing = spacing[spacing > 0]
        tol = float(np.median(spacing)) if spacing.size else 180.0

    indices, matched = [], []
    for want in np.atleast_1d(np.asarray(angles_deg, dtype=float)):
        # Compare on the circle: 179° and -179° are 2° apart.
        delta = np.abs((angles[have] - want + 180.0) % 360.0 - 180.0)
        best = int(np.argmin(delta))
        if delta[best] > tol:
            continue
        idx = int(have[best])
        if idx in indices:
            continue
        indices.append(idx)
        matched.append(float(angles[idx]))

    return np.asarray(indices, dtype=int), np.asarray(matched, dtype=float)
