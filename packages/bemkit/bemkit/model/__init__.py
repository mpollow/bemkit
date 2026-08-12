"""Solver-neutral description of a BEM problem.

Pure numpy data with no I/O and no solver knowledge. Everything else in bemkit
imports this; it imports nothing of ours back.
"""

from bemkit.model.mesh import Medium, Mesh

__all__ = ["Mesh", "Medium"]
