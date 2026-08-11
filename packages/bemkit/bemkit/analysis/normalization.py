"""Shared normalization formula for pressure and velocity quantities."""

import numpy as np


class NoVelocityDrive(ValueError):
    """Raised when the ``acc`` normalization is asked of a model that has none.

    ``acc`` divides by the vibrating area, so it only means anything for a
    model driven by a velocity boundary condition. A scattering model — a plane
    wave or point source against a rigid or absorbing surface — has no such
    area, and NumCalc's raw output is already the meaningful quantity there.
    """


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
    A_vib : float
        Vibrating surface area in m^2.

    Returns
    -------
    (N, nf) or (N, nf, 3) complex ndarray
        Normalized quantity (same shape as input).

    Raises
    ------
    NoVelocityDrive
        If ``A_vib`` is None. This used to warn and hand back the *raw* array,
        which meant a scattering model produced raw Pascals wearing an
        acceleration-normalized label — wrong by a frequency-dependent factor
        and invisible on a plot.
    """
    if A_vib is None:
        raise NoVelocityDrive(
            "acc normalization needs a vibrating area, and this model has no "
            "velocity-driven elements. Use norm='raw' — for a scattering model "
            "(plane wave / point source) that is the meaningful quantity "
            "anyway. If you expected a driven surface, check that the mesh's "
            "vibrating group resolved and that its velocity is non-zero.")

    omega = 2 * np.pi * frequencies
    # Broadcast omega to match input shape: (N, nf) or (N, nf, 3)
    if p_raw.ndim == 2:
        omega_bc = omega[None, :]  # (1, nf)
    else:  # p_raw.ndim == 3
        omega_bc = omega[None, :, None]  # (1, nf, 1)

    return p_raw / (1j * rho0 * omega_bc * A_vib)
