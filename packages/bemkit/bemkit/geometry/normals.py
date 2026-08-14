"""Face and vertex normals — the one place the area-weighting convention lives.

Three consumers need the same quantity for different reasons and must agree:
``metric.curvature`` measures how far each face normal deviates from its
vertices' normals, ``combine.push_clear`` displaces a device along the head's
outward normal, and ``quality`` reasons about orientation. Computing it three
times invites three slightly different conventions.

Area weighting is the choice that matters. An unweighted vertex normal is the
mean of the incident face normals, so a vertex surrounded by one large face and
five slivers is dominated by the slivers — exactly backwards. Weighting by face
area makes the vertex normal follow the surface the faces actually cover, which
is what both curvature estimation and geometric displacement want.

Pure numpy, no I/O.
"""

import numpy as np


def face_normals(vertices, triangles, *, normalise=True):
    """Per-face normals and areas.

    Orientation follows the triangle winding: the normal points along
    ``cross(v1 - v0, v2 - v0)``, so it is outward exactly when the winding is
    counter-clockwise seen from outside (what ``quality.check_orientation``
    verifies).

    Args:
        vertices: (N, 3) coordinates.
        triangles: (F, 3) 0-based connectivity.
        normalise: return unit normals (the default). When False the normals
            keep their ``2 * area`` magnitude, which saves a division for
            callers that only need a direction to dot against.

    Returns:
        ``(normals, areas)`` — (F, 3) and (F,).
    """
    v = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=int)]
    raw = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])     # (F, 3), |raw| = 2A
    areas = np.linalg.norm(raw, axis=1) / 2.0

    if not normalise:
        return raw, areas
    # A degenerate face has no direction to report; leave it at zero rather
    # than dividing by ~0 and emitting a garbage unit vector that would then be
    # area-weighted into its vertices' normals.
    return raw / np.maximum(2.0 * areas, 1e-15)[:, None], areas


def vertex_normals(vertices, triangles):
    """Area-weighted unit vertex normals, (N, 3).

    Vertices touched by no face — or only by degenerate ones — come back as
    zero vectors rather than NaN, so a caller that dots against them gets a
    harmless 0 instead of poisoning an aggregate.
    """
    triangles = np.asarray(triangles, dtype=int)
    n_verts = len(vertices)

    fn, fa = face_normals(vertices, triangles)
    weighted = fa[:, None] * fn                              # (F, 3)

    # Scatter-add each face's contribution to its three corners. triangles
    # ravels as [a0 b0 c0 a1 b1 c1 ...] and the repeat matches it face-for-face
    # — the interleaving has to line up or normals land on the wrong vertices.
    out = np.zeros((n_verts, 3))
    np.add.at(out, triangles.ravel(), np.repeat(weighted, 3, axis=0))

    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-15)


def vertex_areas(vertices, triangles):
    """(N,) sum of incident face areas per vertex — the weights behind
    :func:`vertex_normals`, exposed because curvature normalises by them."""
    triangles = np.asarray(triangles, dtype=int)
    _, fa = face_normals(vertices, triangles, normalise=False)
    return np.bincount(triangles.ravel(), weights=np.repeat(fa, 3),
                       minlength=len(vertices))
