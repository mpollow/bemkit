"""The two leaf types of the model layer: Mesh and Medium.

Pure data. numpy only — no file I/O, no solver knowledge, nothing importable
from the rest of bemkit. That is what makes everything above testable without a
mesh file or a solver binary.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Medium:
    """Acoustic medium. ``c`` in m/s, ``rho`` in kg/m³."""

    c: float = 343.0
    rho: float = 1.2

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.rho <= 0:
            raise ValueError(f"rho must be positive, got {self.rho}")

    @property
    def z0(self):
        """Characteristic impedance rho*c — the normalisation admittance and
        impedance boundary conditions are expressed in."""
        return self.rho * self.c


class Mesh:
    """A triangulated surface, **always in metres**.

    Args:
        vertices: (N, 3) float coordinates in metres.
        triangles: (M, 3) int, 0-based connectivity.
        groups: optional {name: element index array}, e.g. from a mesh file's
            physical groups or a STEP file's face colours.
    """

    def __init__(self, vertices, triangles, groups=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.triangles = np.asarray(triangles, dtype=int)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3), got "
                             f"{self.vertices.shape}")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(f"triangles must be (M, 3), got "
                             f"{self.triangles.shape}")
        if len(self.triangles) and (self.triangles.min() < 0
                                    or self.triangles.max() >= len(self.vertices)):
            raise ValueError(
                f"triangle index out of range: max index "
                f"{self.triangles.max()} but the mesh has "
                f"{len(self.vertices)} vertices")
        self.groups = {}
        for name, idx in (groups or {}).items():
            idx = np.asarray(idx)
            if idx.dtype == bool:
                raise ValueError(
                    f"group {name!r}: boolean masks are not accepted as "
                    f"element indices (silently mis-parses as 0/1 integer "
                    f"indices) — pass np.nonzero(mask)[0] instead")
            idx = idx.astype(int)
            if len(idx) and (idx.min() < 0 or idx.max() >= len(self.triangles)):
                raise ValueError(
                    f"group {name!r}: element index out of range: max "
                    f"index {idx.max()} but the mesh has "
                    f"{len(self.triangles)} elements")
            self.groups[name] = idx

    @property
    def n_nodes(self):
        return len(self.vertices)

    @property
    def n_elements(self):
        return len(self.triangles)

    @property
    def centroids(self):
        """(M, 3) element centroids."""
        return self.vertices[self.triangles].mean(axis=1)

    @property
    def areas(self):
        """(M,) element areas in m²."""
        v = self.vertices[self.triangles]
        return 0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    def __repr__(self):
        return (f"Mesh({self.n_nodes} nodes, {self.n_elements} elements, "
                f"groups={list(self.groups)})")
