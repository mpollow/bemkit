"""Solver-neutral description of a BEM problem.

Pure numpy data with no I/O and no solver knowledge. Everything else in bemkit
imports this; it imports nothing of ours back.

    Mesh  ->  Model(mesh, frequencies, medium, bcs, sources, fields)  ->  Solver
"""

from bemkit.model.conditions import (
    Admittance, Condition, Impedance, Pressure, Velocity,
)
from bemkit.model.fields import circle, fft_grid, on_axis, plane
from bemkit.model.mesh import Medium, Mesh
from bemkit.model.problem import BC, Model, ResolvedBC
from bemkit.model.select import Elements, Group, Near, Selector, Where
from bemkit.model.sources import PlaneWave, PointSource

__all__ = [
    "Mesh", "Medium", "Model", "BC", "ResolvedBC",
    "Selector", "Group", "Elements", "Near", "Where",
    "Condition", "Velocity", "Pressure", "Admittance", "Impedance",
    "PlaneWave", "PointSource",
    "fft_grid", "on_axis", "circle", "plane",
]
