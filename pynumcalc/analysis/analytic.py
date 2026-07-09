"""Analytic solution for spherical cap vibration via SH expansion."""
import numpy as np


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
    try:
        import sphharm as sh
    except ImportError:
        raise ImportError(
            "sphharm is required for analytic solutions. "
            "Install it with: pip install pynumcalc[analytic]"
        ) from None

    r0    = radius_mm / 1e3
    n_vec = np.arange(nmax + 1)

    # Cap velocity SH coefficients (axisymmetric, m=0 only)
    capSH = sh.sphericalcap(nmax, theta=np.radians(angle_deg), SN3D=False, all=False)

    # Y_n0 at each eval point
    Y   = sh.spherical_harmonics_williams(nmax, eval_points)
    Ym0 = Y[:, n_vec**2 + n_vec]                  # (n, nmax+1)
    r   = np.linalg.norm(eval_points, axis=1)      # (n,)

    pressure = np.zeros((len(eval_points), len(frequencies)), dtype=complex)
    for i, f in enumerate(frequencies):
        if f == 0:
            continue
        k  = 2 * np.pi * f / c0
        qm = capSH * (-1j * rho0 * c0) / sh.hn2(n_vec, k * r0, derivative=True)
        pressure[:, i] = (Ym0 * sh.hn2(n_vec, k * r[:, None]) * qm).sum(axis=1)

    return pressure
