"""The acceleration normalization, expressed as a scale factor.

NumCalc solves a unit *velocity* boundary condition and reports pressure in
Pascals for that drive. Dividing by ``j * rho0 * omega * A_vib`` re-expresses
it per unit volume acceleration, which is what makes two models with different
driven areas comparable and what every acc-labelled plot has always shown.

It is a per-frequency scale factor and nothing more, so it belongs on the
linearity seam ``Results.scaled_by`` already provides rather than in a
bespoke array transform: a normalization and a driver response compose, and
neither needs a re-solve.
"""

import numpy as np


class NoVelocityDrive(ValueError):
    """The acc normalization was asked of a model that has no velocity drive.

    ``acc`` divides by the vibrating area, so it only means anything for a
    velocity-driven model. A scattering model — a plane wave or point source
    against a rigid or absorbing surface — has no such area, and NumCalc's raw
    output is already the meaningful quantity there.
    """


def acceleration_scale(model):
    """(nf,) complex spectrum for ``Results.scaled_by`` giving acc units.

    ``1 / (j * rho0 * omega * A_vib)``, evaluated on ``model.frequencies``.

    Raises:
        NoVelocityDrive: the model has no velocity-driven elements, so there
            is no ``A_vib``. Deliberately an error and not a silent fallback
            to raw: raw Pascals under an acc label are wrong by a
            frequency-dependent factor and look perfectly plausible on a plot.
    """
    area = model.vibrating_area
    if area is None:
        raise NoVelocityDrive(
            "acc normalization needs a vibrating area, and this model has no "
            "velocity-driven elements. Use norm='raw' — for a scattering "
            "model (plane wave / point source) that is the meaningful "
            "quantity anyway. If you expected a driven surface, check that "
            "the boundary condition's selector resolved and that its "
            "velocity is non-zero (Velocity(0) is sound-hard).")
    omega = 2.0 * np.pi * np.asarray(model.frequencies, dtype=float)
    return 1.0 / (1j * model.medium.rho * omega * area)
