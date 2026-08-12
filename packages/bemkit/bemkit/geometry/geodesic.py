"""Geodesic (over-the-surface) distance for mesh sizing fields.

GMSH's ``Distance`` field measures straight-line distance through space. For
acoustic BEM that is wrong wherever the geometry folds back on itself: a point
on the outer wall of a horn is millimetres from the piston through the air, but
the surface path runs all the way around the horn mouth. Sizing by the ambient
distance over-refines the entire outer shell for no accuracy gain.

This module computes distance *along* a triangulated surface instead, and maps
it to element sizes with the same linear ramp GMSH's ``Threshold`` field uses.
The result is fed back to GMSH as a background mesh view (see
``bemkit.geometry.generate``).

Distances come from multi-source Dijkstra on the mesh edge graph. Paths are
constrained to follow edges, so they overestimate the true geodesic by a few
percent on a well-shaped mesh (metrication error). That is deliberate: the
result feeds a smooth size ramp and is then approximated again by the mesher,
so the error is invisible in the mesh it produces, and Dijkstra needs no
dependency beyond scipy. If artifacts ever appear, the heat method (Crane et
al.) drops in behind :func:`geodesic_distance` unchanged.

No GMSH import here — the geometry is plain arrays, which keeps this testable
without a mesher.
"""

import numpy as np


def geodesic_distance(nodes, triangles, source_node_indices):
    """Distance from a source node set to every node, measured over the surface.

    Args:
        nodes: (N, 3) vertex coordinates
        triangles: (T, 3) integer vertex indices
        source_node_indices: indices of the nodes to measure from

    Returns:
        (N,) float64 distances. Nodes in a connected component that contains no
        source are ``inf`` — callers decide what that means (see
        :func:`size_from_distance`, which maps ``inf`` to the coarsest size).

    Raises:
        ValueError: if the source set is empty.
    """
    from scipy.sparse import csr_matrix          # lazy: scipy is an extra
    from scipy.sparse.csgraph import dijkstra

    nodes = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    src = np.asarray(source_node_indices, dtype=np.int64).ravel()

    if src.size == 0:
        raise ValueError("geodesic_distance: empty source node set")

    # Deduplicate edges before building the graph. This is correctness-critical,
    # not an optimization: COO construction *sums* duplicate (row, col) entries,
    # so an interior edge contributed by both its triangles would get twice its
    # true length.
    edges = np.vstack([triangles[:, [0, 1]],
                       triangles[:, [1, 2]],
                       triangles[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    weights = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)

    # Both directions explicitly, so the matrix is symmetric regardless of how
    # scipy interprets `directed=False`.
    n = len(nodes)
    graph = csr_matrix(
        (np.r_[weights, weights],
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n, n))

    # min_only collapses all sources into one (N,) result in a single pass.
    # Without it the result is (len(src), N) — a memory blow-up for a source
    # region of any real size.
    return dijkstra(graph, directed=False, indices=src, min_only=True)


def size_from_distance(d, lc_min, lc_max, dist_min, dist_max):
    """Map distance to element size with GMSH's ``Threshold`` ramp.

    ``lc_min`` up to ``dist_min``, linear to ``lc_max`` at ``dist_max``, clamped
    at both ends. Infinite distances (nodes unreachable from the source) fall
    out of the clamp as ``lc_max``, which is the right default: an unreachable
    region is as far from the source as it is possible to be.

    A degenerate ramp (``dist_min == dist_max``) becomes a step function rather
    than a division by zero — GMSH's own Threshold field tolerates this, so this
    reimplementation has to as well.

    Args:
        d: (N,) distances, may contain ``inf``
        lc_min, lc_max: element size at/below dist_min and at/above dist_max
        dist_min, dist_max: the ramp bounds

    Returns:
        (N,) float64 element sizes.
    """
    d = np.asarray(d, dtype=np.float64)

    if dist_max <= dist_min:
        return np.where(d <= dist_min, lc_min, lc_max).astype(np.float64)

    t = np.clip((d - dist_min) / (dist_max - dist_min), 0.0, 1.0)
    return lc_min + t * (lc_max - lc_min)


def pack_scalar_triangles(nodes, triangles, values):
    """Pack a per-node scalar field into GMSH's list-based ``"ST"`` layout.

    GMSH wants coordinate-major data — all three x's, then all three y's, then
    all three z's, then the three values — giving 12 doubles per triangle:

        [x1 x2 x3  y1 y2 y3  z1 z2 z3  v1 v2 v3, <next triangle>, ...]

    Args:
        nodes: (N, 3) vertex coordinates
        triangles: (T, 3) integer vertex indices
        values: (N,) per-node scalars

    Returns:
        (12*T,) float64, flat, ready for ``gmsh.view.addListData``.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    corners = nodes[triangles]              # (T, 3 vertices, 3 coordinates)
    # Swap the last two axes to go from vertex-major (x1 y1 z1 x2 ...) to the
    # coordinate-major order GMSH expects. Getting this wrong yields a garbage
    # field that GMSH accepts without complaint.
    coords = corners.transpose(0, 2, 1).reshape(len(triangles), 9)

    return np.concatenate([coords, values[triangles]], axis=1).ravel()
