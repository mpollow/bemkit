"""Fieldpoint generation.

Functions:
    make_fields_onaxis  — points along a single axis
    make_fields_polar   — points on a circle in a plane
    make_fields_plane   — grid of points on a plane
"""

import numpy as np


def make_fields_onaxis(dim='x', radii=None):
    """Generate points along one axis at specified radii (meters).

    Args:
        dim: Axis character ('x', 'y', or 'z')
        radii: List of distances along axis in meters. Default [-1, -0.1, 0, 0.1, 1].

    Returns:
        (N, 3) array of points
    """
    if radii is None:
        radii = [-1.0, -0.1, 0.0, 0.1, 1.0]
    radii = np.asarray(radii, dtype=float)
    points = np.zeros((len(radii), 3))
    axis = 'xyz'.index(dim)
    points[:, axis] = radii
    return points


def make_fields_polar(dims='xy', res=5, radius=1.0):
    """Generate points on a circle in a plane.

    Args:
        dims: Two-char plane identifier ('xy', 'xz', or 'yz')
        res: Angular resolution in degrees
        radius: Circle radius in meters

    Returns:
        (N, 3) array of points
    """
    angles = np.arange(0, 360, res)
    theta = np.deg2rad(angles)
    ax0 = 'xyz'.index(dims[0])
    ax1 = 'xyz'.index(dims[1])
    points = np.zeros((len(theta), 3))
    points[:, ax0] = radius * np.cos(theta)
    points[:, ax1] = radius * np.sin(theta)
    return points


def make_fields_plane(dims='xy', res=5, limits=None):
    """Generate a grid of points on a plane.

    Args:
        dims: Two-char plane identifier ('xy', 'xz', or 'yz')
        res: Grid spacing in mm (converted to meters internally)
        limits: [min0, max0, min1, max1] in mm. Default [-200, 200, -200, 200].

    Returns:
        (N, 3) array of points in meters
    """
    if limits is None:
        limits = [-200, 200, -200, 200]
    # Convert mm to meters
    scale = 0.001
    v0 = np.arange(limits[0], limits[1] + res, res) * scale
    v1 = np.arange(limits[2], limits[3] + res, res) * scale
    g0, g1 = np.meshgrid(v0, v1, indexing='ij')
    ax0 = 'xyz'.index(dims[0])
    ax1 = 'xyz'.index(dims[1])
    points = np.zeros((g0.size, 3))
    points[:, ax0] = g0.ravel()
    points[:, ax1] = g1.ravel()
    return points

