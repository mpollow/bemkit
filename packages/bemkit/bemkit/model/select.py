"""Selectors: *where* a boundary condition applies.

Each resolves against a Mesh to a sorted array of element indices. They are
deliberately orthogonal to the conditions in `conditions.py` — four selectors
times four conditions is sixteen combinations from eight small classes, where a
dict grammar needed a new key per capability.

Geometric selectors resolve at solve time, so they survive a remesh: an anchor
is a physical place, not an index into one particular triangulation.
"""

from dataclasses import dataclass

import numpy as np

#: An anchor further than this fraction of the mesh extent from its nearest
#: element is almost always a units or coordinate-frame mistake.
_FAR_FRACTION = 0.05


class Selector:
    """Base class. Subclasses implement resolve(mesh) -> (k,) int array."""

    def resolve(self, mesh):
        raise NotImplementedError


@dataclass(frozen=True)
class Group(Selector):
    """Elements of a named physical group, matched case-insensitively.

    Foreign meshers rarely use bemkit's names — Akabak writes ``SourceDisc`` —
    so matching is by name and never by tag id, which such meshes routinely
    number the other way round.
    """

    name: str

    def resolve(self, mesh):
        for key, idx in mesh.groups.items():
            if key.lower() == self.name.lower():
                return np.sort(np.asarray(idx, dtype=int))
        raise KeyError(
            f"no physical group {self.name!r} in this mesh "
            f"(available: {', '.join(mesh.groups) or 'none'})")


@dataclass(frozen=True)
class Elements(Selector):
    """Literal element indices. Tied to one triangulation — prefer Near/Group."""

    indices: tuple

    def __init__(self, indices):
        arr = np.asarray(indices)
        if arr.dtype == bool:
            raise ValueError(
                "Elements(...): boolean masks are not accepted as element "
                "indices (silently mis-parses as 0/1 integer indices) — "
                "pass np.nonzero(mask)[0] instead")
        object.__setattr__(self, "indices", tuple(int(i) for i in indices))

    def resolve(self, mesh):
        idx = np.sort(np.asarray(self.indices, dtype=int))
        if len(idx) and (idx.min() < 0 or idx.max() >= mesh.n_elements):
            raise ValueError(
                f"element index out of range: the mesh has "
                f"{mesh.n_elements} elements")
        return idx


@dataclass(frozen=True)
class Near(Selector):
    """The nearest element to a point, or every element within ``radius``.

    Point and radius are in metres, like everything else in the model layer.
    """

    point: tuple
    radius: float | None = None

    def __init__(self, point, radius=None):
        object.__setattr__(self, "point", tuple(float(x) for x in point))
        object.__setattr__(self, "radius", None if radius is None else float(radius))

    def resolve(self, mesh):
        centroids = mesh.centroids
        d = np.linalg.norm(centroids - np.asarray(self.point), axis=1)
        if self.radius is not None:
            idx = np.nonzero(d <= self.radius)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Near{self.point} radius={self.radius}: no element "
                    f"centroid within it (nearest is {d.min():.4g} m away). "
                    f"Check the units and coordinate frame.")
            return np.sort(idx)

        extent = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
        if extent and d.min() > _FAR_FRACTION * extent:
            raise ValueError(
                f"Near{self.point} is {d.min():.4g} m from its nearest element "
                f"(> {_FAR_FRACTION:.0%} of the mesh extent {extent:.4g} m). "
                f"Check the units and coordinate frame.")
        return np.array([int(np.argmin(d))])


@dataclass(frozen=True)
class Where(Selector):
    """Elements whose centroid satisfies a predicate.

    The predicate takes the (M, 3) centroid array and returns an (M,) bool
    mask. Not serialisable — a run snapshot records the *resolved* indices, and
    model.py is the record of intent.
    """

    predicate: object

    def resolve(self, mesh):
        n = mesh.n_elements
        mask = np.asarray(self.predicate(mesh.centroids))
        if mask.dtype != bool:
            raise TypeError(
                f"Where(...): predicate must return a boolean mask, got "
                f"dtype {mask.dtype} (an index array is silently "
                f"misinterpreted as a mask) — return c[:, i] > x, not indices")
        if mask.shape != (n,):
            raise ValueError(
                f"Where(...): predicate must return a mask of shape ({n},) "
                f"matching the mesh's element count, got {mask.shape}")
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            raise ValueError("Where(...) selected no elements")
        return np.sort(idx)
