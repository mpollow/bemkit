"""Incident sources.

A plane wave is defined by the *direction* it travels in; a point source by its
*position*. Same three numbers, different meaning — a row copied between them
is accepted and silently wrong, which is why they are separate types here
rather than one type with a mode flag.

Amplitudes are scalars: frequency-dependent excitation is post-scaling (see
conditions.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaneWave:
    """Incident plane wave. ``direction`` need not be a unit vector."""

    direction: tuple
    amplitude: complex = 1.0

    def __init__(self, direction, amplitude=1.0):
        d = tuple(float(x) for x in direction)
        if len(d) != 3:
            raise ValueError(f"direction must be 3 numbers, got {direction!r}")
        if not any(d):
            raise ValueError(
                "plane-wave direction is all zeros — it is a direction of "
                "travel, not a position, and it gets normalised")
        object.__setattr__(self, "direction", d)
        object.__setattr__(self, "amplitude", complex(amplitude))


@dataclass(frozen=True)
class PointSource:
    """Monopole at ``position`` (metres).

    Keep it at least one average element edge length off the surface: the
    free-field Green's function is singular at zero distance (arXiv:2312.00005,
    Hint 3).
    """

    position: tuple
    amplitude: complex = 1.0

    def __init__(self, position, amplitude=1.0):
        p = tuple(float(x) for x in position)
        if len(p) != 3:
            raise ValueError(f"position must be 3 numbers, got {position!r}")
        object.__setattr__(self, "position", p)
        object.__setattr__(self, "amplitude", complex(amplitude))
