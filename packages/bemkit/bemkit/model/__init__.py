"""Solver-neutral description of a BEM problem.

Pure numpy data with no I/O and no solver knowledge. Everything else in bemkit
imports this; it imports nothing of ours back.

    Mesh  ->  Model(mesh, frequencies, medium, bcs, sources, fields)  ->  Solver

The one exception to "numpy only" is :mod:`bemkit.model.analytic`, which needs
scipy for its spherical-harmonic machinery. It is reached through a lazy
``__getattr__`` below rather than imported here, so ``import bemkit.model``
stays constructible with nothing but numpy installed — the property that lets
a Model be built and compared without a solver, a mesher or a display stack.
"""

import importlib

from bemkit.model.conditions import (
    Admittance, Condition, Impedance, Pressure, Velocity,
)
from bemkit.model.fields import (
    circle, fft_grid, on_axis, plane, resolve_bins,
)
from bemkit.model.mesh import Medium, Mesh
from bemkit.model.problem import BC, Model, ResolvedBC
from bemkit.model.select import Elements, Group, Near, Selector, Where
from bemkit.model.sources import PlaneWave, PointSource

__all__ = [
    "Mesh", "Medium", "Model", "BC", "ResolvedBC",
    "Selector", "Group", "Elements", "Near", "Where",
    "Condition", "Velocity", "Pressure", "Admittance", "Impedance",
    "PlaneWave", "PointSource",
    "fft_grid", "on_axis", "circle", "plane", "resolve_bins",
    "sphcap_pressure",
]

#: Name -> module, resolved on first access. Keeps scipy out of the import
#: path of everything that does not actually want an analytic reference.
_LAZY = {"sphcap_pressure": "bemkit.model.analytic"}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'bemkit.model' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
