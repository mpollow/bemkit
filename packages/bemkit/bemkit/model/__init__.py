"""Solver-neutral description of a BEM problem.

Pure numpy data with no I/O and no solver knowledge. Everything else in bemkit
imports this; it imports nothing of ours back.
"""

from bemkit.model.conditions import (
    Admittance, Condition, Impedance, Pressure, Velocity,
)
from bemkit.model.mesh import Medium, Mesh
from bemkit.model.select import Elements, Group, Near, Selector, Where

__all__ = ["Mesh", "Medium",
           "Selector", "Group", "Elements", "Near", "Where",
           "Condition", "Velocity", "Pressure", "Admittance", "Impedance"]
