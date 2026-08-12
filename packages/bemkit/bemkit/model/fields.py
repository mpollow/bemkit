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


def resolve_bins(frequencies, *, freq=None, bins=None, preview=None):
    """Resolve a partial-band selection to sorted unique 1-based bin indices.

    A partial run computes a subset of the same grid a full run would, which is
    what lets its results merge into a later full run. Requested frequencies are
    therefore snapped onto that grid, never honoured exactly. Exactly one
    selector must be given.

    Args:
        frequencies: The model's frequency array in Hz — ``ModelConfig
            .freq_array()``. Taken as data rather than derived from sr/nbins,
            so a non-uniform ``[frequencies]`` band selects the same way.
        freq: Sequence of frequencies in Hz, each snapped to the nearest bin.
            Ties resolve to the lower bin. A frequency that is negative,
            non-finite, or above sr/2 raises rather than snapping silently.
        bins: Sequence of literal 1-based bin indices.
        preview: N bins spread logarithmically over the band. Log spacing
            because a linear pick from a linear grid says nothing about the
            bottom octaves; preview=1 resolves to the lowest bin. Bins that
            coincide after snapping collapse, so the result may be shorter
            than N.

    Returns:
        Sorted, unique, 1-based bin indices.
    """
    given = [name for name, val in
             (("freq", freq), ("bins", bins), ("preview", preview))
             if val is not None]
    if len(given) != 1:
        raise ValueError(
            f"resolve_bins needs exactly one of freq/bins/preview, got "
            f"{', '.join(given) if given else 'none'}")

    freqs = np.asarray(frequencies, dtype=float)
    nf = len(freqs)
    if nf == 0:
        raise ValueError(
            "no frequency bins to select from (a model needs at least one "
            "simulated frequency)")

    if bins is not None:
        wanted = np.asarray(list(bins), dtype=float)
        if wanted.size == 0:
            raise ValueError("bins selector is empty")
        # ~isfinite is part of the mask, not a separate check: floor(inf) == inf,
        # so inf would pass an integrality test and then cast to INT_MIN, and
        # the range error below would name -9223372036854775808.
        non_integral = wanted[~np.isfinite(wanted) | (wanted != np.floor(wanted))]
        if non_integral.size:
            raise ValueError(
                f"bin indices must be whole numbers: "
                f"{', '.join(f'{v:g}' for v in non_integral)}")
        wanted_bins = wanted.astype(int)
        outside = wanted_bins[(wanted_bins < 1) | (wanted_bins > nf)]
        if outside.size:
            raise ValueError(
                f"outside the band: {', '.join(str(b) for b in outside)} — "
                f"this model has {nf} bins, valid indices 1..{nf} "
                f"(bins are 1-based, like NumCalc's -istart)")
        return sorted(set(wanted_bins.tolist()))

    if preview is not None:
        n = int(preview)
        if n < 1:
            raise ValueError(f"preview needs at least 1 bin, got {n}")
        wanted_hz = np.geomspace(freqs[0], freqs[-1], n)
    else:
        wanted_hz = np.asarray([float(f) for f in freq], dtype=float)
        if wanted_hz.size == 0:
            raise ValueError("freq selector is empty")
        bad_mask = ~np.isfinite(wanted_hz) | (wanted_hz < 0)
        if bad_mask.any():
            bad = wanted_hz[bad_mask]
            raise ValueError(
                f"not valid as a frequency (must be finite and non-negative): "
                f"{', '.join(f'{v:g}' for v in bad)}")
        above = wanted_hz[wanted_hz > freqs[-1]]
        if above.size:
            raise ValueError(
                f"above the simulated band: "
                f"{', '.join(f'{v:g}' for v in above)} Hz — the band reaches "
                f"{freqs[-1]:g} Hz")

    return sorted({int(np.argmin(np.abs(freqs - f))) + 1 for f in wanted_hz})
