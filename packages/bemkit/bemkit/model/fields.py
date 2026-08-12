"""Field-point generators.

Every generator returns a plain (N, 3) array of metres. Field points are a
``{name: array}`` dict on the Model — there is no schema, no file discovery and
no exec hook, because ``np.load("scan.npy")`` already does the only thing such
a feature would have added.
"""

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}


def fft_grid(sr, nbins):
    """The uniform FFT grid, DC excluded — shape (nbins-1,).

    Needed when a time-domain result is wanted; any other ascending array is a
    perfectly good band for the frequency domain.
    """
    return np.linspace(0, sr / 2, nbins)[1:]


def on_axis(axis, distances):
    """Points along one axis at the given distances (metres)."""
    if axis not in _AXES:
        raise ValueError(f"axis must be one of x, y, z — got {axis!r}")
    distances = np.asarray(distances, dtype=float)
    pts = np.zeros((len(distances), 3))
    pts[:, _AXES[axis]] = distances
    return pts


def circle(dims, res=5.0, radius=1.0):
    """A circle of radius ``radius`` in the plane spanned by ``dims``.

    ``res`` is the angular step in degrees. ``dims="zx"`` puts 0 degrees on +z
    and +90 on +x. When ``res`` does not divide 360 evenly, the last step
    before closing the loop is shorter than the rest (matching NumCalc's
    polar-fieldpoint convention) — the loop still closes on 0 degrees, it is
    just not evenly spaced all the way round.
    """
    if len(dims) != 2 or any(d not in _AXES for d in dims):
        raise ValueError(f"dims must be two of x, y, z — got {dims!r}")
    if res <= 0:
        raise ValueError(f"res must be positive, got {res!r}")
    ang = np.deg2rad(np.arange(0.0, 360.0, res))
    pts = np.zeros((len(ang), 3))
    pts[:, _AXES[dims[0]]] = radius * np.cos(ang)
    pts[:, _AXES[dims[1]]] = radius * np.sin(ang)
    return pts


def plane(dims, res=0.05, limits=(-0.5, 0.5, -0.5, 0.5)):
    """A regular grid on the plane spanned by ``dims``. All metres.

    ``res`` need not divide the span evenly: the grid starts at the lower
    limit and stops at the last multiple of ``res`` that still fits *inside*
    the requested bounds, rather than overshooting past the upper limit by up
    to half a step.
    """
    if len(dims) != 2 or any(d not in _AXES for d in dims):
        raise ValueError(f"dims must be two of x, y, z — got {dims!r}")
    if res <= 0:
        raise ValueError(f"res must be positive, got {res!r}")
    a0, a1, b0, b1 = limits
    # A small relative tolerance (not res/2) absorbs float round-off in the
    # endpoint without admitting a point outside [a0, a1].
    tol = res * 1e-9
    a = np.arange(a0, a1 + tol, res)
    b = np.arange(b0, b1 + tol, res)
    A, B = np.meshgrid(a, b, indexing="ij")
    pts = np.zeros((A.size, 3))
    pts[:, _AXES[dims[0]]] = A.ravel()
    pts[:, _AXES[dims[1]]] = B.ravel()
    return pts
