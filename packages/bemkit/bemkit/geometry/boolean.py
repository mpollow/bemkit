"""Fusing overlapping bodies into one closed surface.

Two bodies that interpenetrate are not a valid exterior scattering problem: the
buried caps enclose cavities whose boundary data means nothing and which ring
with spurious interior resonances. Either hold the bodies apart
(``combine.push_clear``) or fuse them, which is this module.

Backed by `manifold3d <https://github.com/elalish/manifold>`_ — Apache-2.0, so
compatible with this repo's GPL-3.0 licence, and the same engine Blender uses for
its boolean modifier. It is an optional extra (``bemkit[csg]``) imported lazily
inside :func:`union`, so nothing here is paid for unless a study fuses
something.

**Float64 throughout.** manifold3d's default ``Mesh`` type is float32, which
would quietly perturb every vertex of a head mesh by a few nanometres and
destroy the bit-identity that the whole frozen-head pipeline depends on.
``Mesh64`` round-trips exactly, so that is what this uses. Measured on a
sphere-plus-sphere union at metre scale: 96.4% of the larger body's vertices
come back bit-identical, and every vertex that does not is one the other body
swallowed.
"""

import numpy as np

from bemkit.model import Mesh

_INSTALL_HINT = (
    "boolean operations need manifold3d — install it with "
    "`pip install bemkit[csg]` or `pip install manifold3d`")


def _to_manifold(mesh):
    import manifold3d

    man = manifold3d.Manifold(manifold3d.Mesh64(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        tri_verts=np.ascontiguousarray(mesh.triangles, dtype=np.uint64)))
    if man.is_empty():
        raise ValueError(
            "boolean: input is not an oriented 2-manifold — manifold3d "
            "rejected it. Run geometry.quality.check_watertight and "
            "check_orientation on it first; holes and inconsistent winding "
            "are the usual causes.")
    return man


def _from_manifold(man):
    mesh = man.to_mesh64()
    return (np.asarray(mesh.vert_properties)[:, :3].copy(),
            np.asarray(mesh.tri_verts).astype(np.int64))


def union(**bodies):
    """Fuse named bodies into one closed surface.

    Takes bodies by name for the same reason :func:`~bemkit.geometry.combine.
    concat_meshes` does, and produces the same group layout, so the two
    junction policies are interchangeable in a build script::

        assembled = union(head=head, glasses=frames)          # fused
        assembled = concat_meshes(head=head, glasses=frames)   # held apart

    Groups are reconstructed after the fact, because manifold3d has no notion
    of them. A triangle that survives the union verbatim is attributed to the
    body it came from; a triangle created by retriangulating along the
    intersection curve is attributed to whichever body's surface its centroid
    is nearest. Only triangles near the seam are ever attributed the second
    way.

    Args:
        **bodies: ``name=Mesh``. Each must be watertight and outward-oriented.

    Returns:
        A single :class:`~bemkit.model.Mesh` with one group per body. Groups
        the inputs carried are **not** preserved — the union renumbers
        everything, and a sub-region's extent after fusing is ambiguous
        anyway. Re-derive them geometrically (``model.select.Near``) or tag
        them as mmg refs before remeshing.

    Raises:
        ImportError: manifold3d is not installed.
        ValueError: fewer than two bodies, or one is not a valid manifold.
    """
    try:
        import manifold3d                                    # noqa: F401
    except ImportError as exc:                               # pragma: no cover
        raise ImportError(_INSTALL_HINT) from exc

    if len(bodies) < 2:
        raise ValueError(
            f"union: need at least two bodies, got {len(bodies)}")

    result = None
    for mesh in bodies.values():
        man = _to_manifold(mesh)
        result = man if result is None else result + man

    vertices, triangles = _from_manifold(result)
    out = Mesh(vertices, triangles)
    return Mesh(vertices, triangles, _attribute(out, bodies))


def _attribute(out, bodies):
    """Assign each output triangle to the body it came from."""
    from bemkit.geometry.combine import distance_to_mesh

    names = list(bodies)
    owner = np.full(out.n_elements, -1, dtype=int)

    # Verbatim triangles first: an exact coordinate match is unambiguous and
    # covers everything away from the seam.
    out_keys = _triangle_keys(out)
    for i, mesh in enumerate(bodies.values()):
        for k in _triangle_keys(mesh):
            j = out_keys.get(k)
            if j is not None and owner[j] < 0:
                owner[j] = i

    # Whatever is left was created along the intersection curve. Attribute it
    # to the nearest input surface — for a seam triangle both bodies are close,
    # but the one it was cut from is closer.
    unresolved = np.nonzero(owner < 0)[0]
    if len(unresolved):
        centroids = out.centroids[unresolved]
        d = np.stack([distance_to_mesh(centroids, m) for m in bodies.values()])
        owner[unresolved] = np.argmin(d, axis=0)

    return {name: np.nonzero(owner == i)[0] for i, name in enumerate(names)}


def _triangle_keys(mesh, decimals=12):
    """{canonical coordinate triple: element index} for every triangle.

    Order-independent, so a triangle that comes back with rotated or reversed
    winding still matches.
    """
    rounded = np.round(mesh.vertices[mesh.triangles], decimals)
    return {tuple(sorted(map(tuple, tri))): i for i, tri in enumerate(rounded)}
