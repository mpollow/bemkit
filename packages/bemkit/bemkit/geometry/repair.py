"""Closing holes in a surface that has to be a BEM boundary.

A BEM boundary must be closed: the formulation divides space into an inside and
an outside, and a surface with a hole does neither. CAD exports routinely leave
one anyway — a sound port modelled as an opening, a face suppressed for
manufacturing — and every downstream tool then fails somewhere unhelpful.
manifold3d simply returns an empty result; NumCalc solves it and returns
nonsense.

The fill here is deliberately minimal: each boundary loop gets one new vertex at
its centroid and a fan of triangles back to the loop. That is exactly right for
a small, roughly planar opening — a nozzle aperture, a bolt hole, a suppressed
fillet — and exactly wrong for a large or strongly non-planar boundary, where a
fan from a single centroid produces a cone or a self-intersecting mess rather
than the surface that ought to be there. :func:`boundary_loops` reports the size
and planarity of every loop so the caller can tell the two cases apart, and
:func:`fill_holes` refuses a loop it judges too non-planar unless told
otherwise.

This is not a general mesh-repair library. It closes simple holes and says so
loudly when a hole is not simple.
"""

import numpy as np


def _boundary_edges(triangles):
    """Directed boundary edges, ``(k, 2)``.

    An edge on a closed manifold is traversed once in each direction by the two
    faces sharing it. An edge traversed only once is a boundary, and the
    direction it *was* traversed in is what tells us how to wind the patch that
    closes it.
    """
    directed = np.vstack([triangles[:, [0, 1]],
                          triangles[:, [1, 2]],
                          triangles[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True,
                                   return_counts=True)
    return directed[counts[inverse.ravel()] == 1]


def boundary_loops(mesh):
    """Describe every hole in ``mesh``.

    Returns:
        List of dicts, largest loop first, each with:

        * ``edges`` — ``(k, 2)`` directed boundary edges of this loop
        * ``vertices`` — the vertex indices around it
        * ``centroid`` — mean position, metres
        * ``span`` — bounding-box extent, metres
        * ``planarity`` — RMS distance of the loop from its best-fit plane,
          divided by the loop's radius. Near 0 is a flat opening a fan will
          close cleanly; above ~0.1 the loop is a saddle or a slot and a fan
          fill will not resemble the missing surface.

        Empty when the mesh is already closed.
    """
    edges = _boundary_edges(np.asarray(mesh.triangles, dtype=int))
    if not len(edges):
        return []

    # Group the boundary edges into loops: connected components of the graph
    # they form. Each component is one hole.
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    seen, loops = set(), []
    for start in adjacency:
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        loops.append(set(component))

    out = []
    for loop in loops:
        mask = np.array([int(a) in loop for a in edges[:, 0]])
        pts = mesh.vertices[sorted(loop)]
        centroid = pts.mean(axis=0)

        centred = pts - centroid
        # Smallest singular direction is the best-fit plane normal; the
        # residual along it is how far the loop departs from flat.
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        radius = np.linalg.norm(centred, axis=1).max()
        rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))

        out.append({"edges": edges[mask],
                    "vertices": np.array(sorted(loop)),
                    "centroid": centroid,
                    "span": pts.max(axis=0) - pts.min(axis=0),
                    "planarity": rms / radius if radius > 0 else 0.0})

    return sorted(out, key=lambda d: -len(d["vertices"]))


def fill_holes(mesh, *, max_planarity=0.1, max_vertices=None):
    """Cap every boundary loop with a centroid fan.

    Winding is taken from the boundary itself: a boundary edge was traversed
    ``a -> b`` by its one adjacent face, so the triangle that closes it runs
    ``b -> a -> centroid``. That makes the patch agree with the surface it is
    sewn onto, whichever way round the surface happens to be — no assumption
    that the input is outward-oriented.

    Args:
        mesh: a :class:`~bemkit.model.Mesh` with holes.
        max_planarity: refuse a loop whose out-of-plane RMS exceeds this
            fraction of its radius (see :func:`boundary_loops`). Pass ``None``
            to fill regardless.
        max_vertices: refuse a loop with more vertices than this. A large
            boundary is usually a missing region rather than a hole, and
            deserves a modelling decision instead of a fan.

    Returns:
        A new Mesh with the holes closed. Existing vertices and triangles are
        untouched and keep their indices — the fill is appended — so groups
        stay valid and anything already frozen stays frozen.

    Raises:
        ValueError: a loop exceeds ``max_planarity`` or ``max_vertices``,
            naming the numbers so the caller can judge.
    """
    from bemkit.model import Mesh

    loops = boundary_loops(mesh)
    if not loops:
        return Mesh(mesh.vertices, mesh.triangles, mesh.groups)

    for i, loop in enumerate(loops):
        n = len(loop["vertices"])
        if max_vertices is not None and n > max_vertices:
            raise ValueError(
                f"fill_holes: loop {i} has {n} vertices, over the "
                f"max_vertices={max_vertices} limit — a boundary that large is "
                f"usually a missing region, not a hole")
        if max_planarity is not None and loop["planarity"] > max_planarity:
            raise ValueError(
                f"fill_holes: loop {i} is not planar enough to fan-fill "
                f"(out-of-plane RMS is {loop['planarity']:.3f} of its radius, "
                f"limit {max_planarity}). A centroid fan would invent a cone "
                f"here. Fill it in a mesher, or pass max_planarity=None if a "
                f"fan really is what you want.")

    vertices = [np.asarray(mesh.vertices, dtype=float)]
    triangles = [np.asarray(mesh.triangles, dtype=int)]
    next_index = mesh.n_nodes

    for loop in loops:
        vertices.append(loop["centroid"][None, :])
        a, b = loop["edges"][:, 0], loop["edges"][:, 1]
        triangles.append(np.column_stack([b, a, np.full(len(a), next_index)]))
        next_index += 1

    return Mesh(np.vstack(vertices), np.vstack(triangles), mesh.groups)
