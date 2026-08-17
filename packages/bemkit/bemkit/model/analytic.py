"""Analytic solution for spherical cap vibration via SH expansion.

The spherical-harmonic helpers below (``_cart2sph``, ``_index_n``, ``_index_m``,
``_hn2``, ``_spherical_harmonics_williams``, ``_sphericalcap``) are vendored from
the ``sphharm`` package (same author) so that analytic references need only
``scipy`` rather than a separate dependency.
"""
import numpy as np

try:
    import scipy.special
except ImportError:
    raise ImportError(
        "scipy is required for analytic solutions. "
        "Install it with: pip install bemkit[analytic]"
    ) from None


# --- vendored sphharm helpers -------------------------------------------------

def _cart2sph(cart):
    """Cartesian (N, 3) -> spherical (N, 3) as [r, theta (colatitude), phi], radians."""
    x, y, z = np.asarray(cart, dtype=float).T
    return np.stack([np.sqrt(x * x + y * y + z * z),
                     np.arctan2(np.hypot(x, y), z),
                     np.arctan2(y, x)], axis=1)


def _index_n(nmax):
    """Order index n for each SH channel up to nmax, in ACN order."""
    return np.sqrt(np.arange((nmax + 1) ** 2)).astype(int)


def _index_m(nmax):
    """Degree index m for each SH channel up to nmax, in ACN order."""
    m = []
    for n in range(nmax + 1):
        m += list(range(-n, n + 1))
    return np.array(m, dtype=int)


def _hn2(n, z, derivative=False):
    """Spherical Hankel function of the second kind (outgoing wave).

    ``y_n`` overflows for ``n`` well past ``z``, and ``1j * inf`` is
    ``nan + inf j``, so the warning here is expected rather than diagnostic.
    :func:`_check_representable` is what turns it into an error a caller can
    act on, so it is silenced rather than emitted twice.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        jn = scipy.special.spherical_jn(n, z, derivative)
        yn = scipy.special.spherical_yn(n, z, derivative)
        return jn - 1j * yn


def _spherical_harmonics_williams(nmax, cart):
    """Complex spherical harmonics up to order nmax (N3D, Condon-Shortley phase).

    Points are Cartesian (N, 3); radial distance is ignored. Returns (N, (nmax+1)**2)
    in ACN order.
    """
    r, t, p = _cart2sph(cart).T
    n, m = _index_n(nmax), _index_m(nmax)
    return scipy.special.sph_harm_y(n, m, t[:, None], p[:, None])


def _zonal_harmonics(nmax, cart):
    """The m=0 column of :func:`_spherical_harmonics_williams`, (N, nmax+1).

    Same scipy call on the same arguments, so the values are identical — but
    an axisymmetric cap only excites m=0, and evaluating the full ACN set to
    keep one channel in every (2n+1) is most of the cost of the whole closed
    form. At nmax=35 that is 36 channels instead of 1296, which is what makes
    scoring a 17k-element boundary solution take a second rather than a minute.
    """
    _, t, p = _cart2sph(cart).T
    n = np.arange(nmax + 1)
    return scipy.special.sph_harm_y(n, 0, t[:, None], p[:, None])


def _sphericalcap(nmax, theta):
    """m=0 SH coefficients of a spherical cap at the north pole (orthonormal, non-SN3D).

    ``theta`` is the half opening angle in radians. Returns the (nmax+1,) coefficients
    for m=0 only (equivalent to ``sphharm.sphericalcap(nmax, theta, SN3D=False, all=False)``).
    """
    x = np.cos(theta)
    pre = np.sqrt(np.pi) * np.sqrt(2 * np.arange(nmax + 1) + 1)
    leg_int = np.zeros(nmax + 1, dtype=np.complex128)
    leg_int[0] = 1 - x
    for n in range(nmax):
        minus = scipy.special.lpmv(0, n, x)
        plus = scipy.special.lpmv(0, n + 2, x)
        leg_int[n + 1] = (minus - plus) / (2 * n + 3)
    return pre * leg_int


# --- analytic solver ----------------------------------------------------------

#: Orders kept above the largest ``k*r`` in play, when ``nmax`` is chosen
#: automatically. The series terms fall off superexponentially past n ~ kr;
#: measured on a 30 degree cap over 250 Hz - 16 kHz, this margin already
#: reproduces nmax = 80 to better than 1e-13 relative.
NMAX_MARGIN = 20


def _auto_nmax(k_max, r_max):
    return int(np.ceil(k_max * r_max)) + NMAX_MARGIN


def sphcap_pressure(eval_points, frequencies, c0, rho0, angle_deg, radius_mm,
                    nmax=None):
    """Pressure field for axisymmetric spherical cap with unit velocity BC.

    Parameters
    ----------
    eval_points : (n, 3) float — field points in metres
    frequencies : (nf,) float — Hz; 0 entries give zero pressure
    c0, rho0    : float — speed of sound (m/s), air density (kg/m³)
    angle_deg   : float — cap half-angle in degrees
    radius_mm   : float — sphere radius in mm
    nmax        : int or None — SH truncation order. ``None`` picks
        ``ceil(k_max * r_max) + NMAX_MARGIN``, which is converged and stays
        inside the range where ``y_n`` is representable.

    Returns
    -------
    pressure : (n, nf) complex128 — pressure in Pa for v=1 BC

    Raises
    ------
    ValueError
        If ``nmax`` is high enough that ``y_n`` overflows to infinity. The
        series does not get more accurate past that point, it gets NaN — and a
        reference that returns NaN when asked for more accuracy is worse than
        one that refuses.
    """
    eval_points = np.asarray(eval_points, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    r0 = radius_mm / 1e3
    r = np.linalg.norm(eval_points, axis=1)      # (n,)

    # Sized on k*r0, not on the field radius: the expansion coefficients are
    # fixed at the sphere, so the number of channels that carry any energy is
    # set there. Orders past k*r0 are what overflow, not what add accuracy.
    k_max = 2 * np.pi * float(np.max(np.abs(frequencies), initial=0.0)) / c0
    if nmax is None:
        nmax = _auto_nmax(k_max, r0)
    n_vec = np.arange(nmax + 1)

    # Cap velocity SH coefficients (axisymmetric, m=0 only)
    capSH = _sphericalcap(nmax, theta=np.radians(angle_deg))

    # Y_n0 at each eval point
    Ym0 = _zonal_harmonics(nmax, eval_points)     # (n, nmax+1)

    pressure = np.zeros((len(eval_points), len(frequencies)), dtype=complex)
    for i, f in enumerate(frequencies):
        if f == 0:
            continue
        k  = 2 * np.pi * f / c0
        surface = _hn2(n_vec, k * r0, derivative=True)
        field = _hn2(n_vec, k * r[:, None])
        _check_representable(surface, field, nmax, k, r0)
        qm = capSH * (-1j * rho0 * c0) / surface
        pressure[:, i] = (Ym0 * field * qm).sum(axis=1)

    return pressure


def _check_representable(surface, field, nmax, k, r0):
    """Reject a truncation order that has run off the end of float64.

    ``y_n(x) ~ -(2n-1)!!/x^(n+1)``, so past roughly ``n = 5x + 20`` it
    overflows. ``1j * inf`` is ``nan + inf j``, so the NaN appears in
    ``_hn2`` itself and then propagates through the whole sum — silently, and
    only at some frequencies, which is how this stayed unnoticed. Truncating
    lower is not a loss: those orders contribute nothing but noise.
    """
    if np.all(np.isfinite(surface)) and np.all(np.isfinite(field)):
        return
    finite = np.isfinite(surface) & np.isfinite(field).all(axis=0)
    bad = int(np.argmin(finite))
    raise ValueError(
        f"sphcap_pressure: the spherical Hankel function is not representable "
        f"at n={bad} (nmax={nmax}, k={k:.4g} 1/m, sphere radius {r0:.4g} m), "
        f"so the series returns NaN rather than a more accurate answer. Pass "
        f"nmax below {bad}, or nmax=None to have {_auto_nmax(k, r0)} chosen.")
