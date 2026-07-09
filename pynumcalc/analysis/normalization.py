"""Shared normalization formula for pressure and velocity quantities."""

import warnings
import numpy as np


def normalize_pressure(p_raw, frequencies, rho0, A_vib):
    """Normalize raw pressure/velocity from velocity BC to acceleration BC.

    Applies the formula: p_acc = p_raw / (1j * rho0 * omega * A_vib)

    Parameters
    ----------
    p_raw : (N, nf) or (N, nf, 3) complex ndarray
        Raw pressure or velocity from NumCalc (velocity boundary condition output).
        Can be 2D (pressure/scalar velocity) or 3D (vector velocity).
    frequencies : (nf,) float ndarray
        Frequency bins in Hz.
    rho0 : float
        Air density in kg/m^3.
    A_vib : float or None
        Vibrating surface area in m^2. If None, returns a copy of p_raw
        with a warning.

    Returns
    -------
    (N, nf) or (N, nf, 3) complex ndarray
        Normalized quantity (same shape as input).
    """
    if A_vib is None:
        warnings.warn(
            "No vibrating elements found — pressure not normalized",
            stacklevel=2,
        )
        return p_raw.copy()

    omega = 2 * np.pi * frequencies
    # Broadcast omega to match input shape: (N, nf) or (N, nf, 3)
    if p_raw.ndim == 2:
        omega_bc = omega[None, :]  # (1, nf)
    else:  # p_raw.ndim == 3
        omega_bc = omega[None, :, None]  # (1, nf, 1)

    return p_raw / (1j * rho0 * omega_bc * A_vib)
