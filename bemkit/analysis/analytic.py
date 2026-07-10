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
    sph = [[np.linalg.norm([x, y, z]),
            np.arctan2(np.linalg.norm([x, y]), z),
            np.arctan2(y, x)]
           for x, y, z in cart]
    return np.array(sph)


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
    """Spherical Hankel function of the second kind (outgoing wave)."""
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

def sphcap_pressure(eval_points, frequencies, c0, rho0, angle_deg, radius_mm, nmax):
    """Pressure field for axisymmetric spherical cap with unit velocity BC.

    Parameters
    ----------
    eval_points : (n, 3) float — field points in metres
    frequencies : (nf,) float — Hz; 0 entries give zero pressure
    c0, rho0    : float — speed of sound (m/s), air density (kg/m³)
    angle_deg   : float — cap half-angle in degrees
    radius_mm   : float — sphere radius in mm
    nmax        : int   — SH truncation order

    Returns
    -------
    pressure : (n, nf) complex128 — pressure in Pa for v=1 BC
    """
    r0    = radius_mm / 1e3
    n_vec = np.arange(nmax + 1)

    # Cap velocity SH coefficients (axisymmetric, m=0 only)
    capSH = _sphericalcap(nmax, theta=np.radians(angle_deg))

    # Y_n0 at each eval point
    Y   = _spherical_harmonics_williams(nmax, eval_points)
    Ym0 = Y[:, n_vec**2 + n_vec]                  # (n, nmax+1)
    r   = np.linalg.norm(eval_points, axis=1)      # (n,)

    pressure = np.zeros((len(eval_points), len(frequencies)), dtype=complex)
    for i, f in enumerate(frequencies):
        if f == 0:
            continue
        k  = 2 * np.pi * f / c0
        qm = capSH * (-1j * rho0 * c0) / _hn2(n_vec, k * r0, derivative=True)
        pressure[:, i] = (Ym0 * _hn2(n_vec, k * r[:, None]) * qm).sum(axis=1)

    return pressure
