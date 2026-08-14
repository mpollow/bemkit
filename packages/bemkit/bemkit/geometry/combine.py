"""Assembling several bodies into one Mesh, and keeping them apart.

A head, a pair of glasses and an earbud arrive as three separate surfaces and
have to become one ``Mesh`` before a solver sees them. Exterior BEM does not
require them to be *fused* — a collection of disjoint closed surfaces is a
perfectly good scatterer — so the cheap assembly is concatenation, and fusing
(``geometry.boolean``) is a separate, optional step.

What concatenation must not do is lose track of which elements came from
which body. That is why :func:`concat_meshes` takes bodies **by name** and has
no positional form: an unnamed merge produces a mesh whose parts can only be
addressed by raw integer index, which is precisely how ``models/glassesonme_v2``
ended up driving ``Elements(15293)`` for a group its name claims is on the
glasses when it actually landed on the head.

The index arithmetic here generalises what ``io.mirror_mesh`` already does for
symmetry reflections; that function stays as it is, since it also has to flip
winding per copy and answer a different question.

Pure numpy apart from two lazy scipy/solver imports for the proximity queries.
"""

import numpy as np

from bemkit.geometry.geodesic import geodesic_distance
from bemkit.geometry.normals import vertex_normals
from bemkit.model import Mesh

#: Separates a body name from a group name it brought with it, as in
#: ``"glasses:frame"``. ``model.select.Group`` matches case-insensitively on the
#: whole string, so a namespaced group is addressed by its full name.
NAMESPACE_SEP = ":"


def concat_meshes(**bodies):
    """Concatenate named bodies into a single Mesh.

    Every input keeps its own triangulation untouched — vertices are stacked in
    argument order and triangle indices offset — so the first body's elements
    are still ``0 .. n-1`` in the result. Put the body you want index-stable
    first (in practice: the head).

    Groups are namespaced to keep two bodies from colliding on a common name
    like ``"rigid"``, and each body additionally gets a group covering all of
    its own elements::

        concat_meshes(head=head, glasses=frames)
        # groups: "head", "glasses",
        #         "head:pinna_left", ... , "glasses:rigid", ...

    Args:
        **bodies: ``name=Mesh``. Order is the call order (Python preserves it).
            A name containing :data:`NAMESPACE_SEP` is rejected — it would make
            the resulting group names ambiguous. Non-identifier names are fine
            via ``concat_meshes(**{"earbud left": m})``.

    Returns:
        A new :class:`~bemkit.model.Mesh`. Inputs are not modified.

    Raises:
        ValueError: no bodies given, or a name contains ``":"``.
    """
    if not bodies:
        raise ValueError("concat_meshes: at least one body is required")
    for name in bodies:
        if NAMESPACE_SEP in name:
            raise ValueError(
                f"concat_meshes: body name {name!r} contains "
                f"{NAMESPACE_SEP!r}, which separates a body from its groups "
                f"— the result would be ambiguous")

    meshes = list(bodies.values())
    # Offset each body's triangle indices by the vertices already stacked
    # before it. Same cumsum-of-preceding-lengths shape as io.mirror_mesh.
    v_starts = np.cumsum([0] + [m.n_nodes for m in meshes[:-1]])
    e_starts = np.cumsum([0] + [m.n_elements for m in meshes[:-1]])

    vertices = np.vstack([m.vertices for m in meshes])
    triangles = np.vstack([m.triangles + s for m, s in zip(meshes, v_starts)])

    groups = {}
    for (name, mesh), e0 in zip(bodies.items(), e_starts):
        groups[name] = np.arange(e0, e0 + mesh.n_elements)
        for gname, idx in mesh.groups.items():
            groups[f"{name}{NAMESPACE_SEP}{gname}"] = np.asarray(idx) + e0

    return Mesh(vertices, triangles, groups)


def weld(mesh, tol=1e-9):
    """Merge vertices that coincide to within ``tol``, keeping element order.

    Triangle *order* is preserved, so element indices and every group survive.
    Only the connectivity changes — which is the point: an STL stores each
    facet independently, and until its corners are welded the surface has no
    shared edges, so every connectivity-based question (:func:`shells`,
    watertightness, geodesic distance) answers nonsense.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        tol: coordinates are snapped to this grid before comparison. The
            default is tight enough to only catch exact duplicates written at
            float precision.

    Returns:
        A new Mesh with duplicate vertices collapsed. Vertices left unreferenced
        by the collapse are dropped.
    """
    if tol <= 0:
        raise ValueError(f"weld: tol must be positive, got {tol}")

    # Snap to a grid, then group identical keys. Rounding rather than a
    # neighbour search means two points either side of a grid line can survive
    # as distinct despite being within tol — acceptable for de-duplicating
    # facet corners, which are bit-identical or genuinely far apart.
    keys = np.round(np.asarray(mesh.vertices) / tol).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)

    # np.unique sorts; keep the original coordinates of each representative
    # rather than the snapped ones, so welding never moves the surface.
    return Mesh(mesh.vertices[first], inverse.ravel()[mesh.triangles],
                mesh.groups)


def shells(mesh, *, tol=None):
    """Split a mesh into connected components, largest first.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        tol: when given, vertices are welded to this tolerance *for the purpose
            of the connectivity analysis only*. Pass this for a mesh read from
            an STL, where facets do not share corners and every triangle would
            otherwise be its own shell. The returned indices always refer to
            ``mesh``'s own elements.

    Returns:
        List of ``(k,) int`` element index arrays, sorted by descending element
        count — so ``shells(head_plus_glasses)[0]`` is the head.
    """
    from scipy.sparse import csr_matrix                # lazy: scipy is an extra
    from scipy.sparse.csgraph import connected_components

    triangles = weld(mesh, tol).triangles if tol else mesh.triangles
    n = int(triangles.max()) + 1 if len(triangles) else 0
    if not n:
        return []

    edges = np.vstack([triangles[:, [0, 1]],
                       triangles[:, [1, 2]],
                       triangles[:, [2, 0]]])
    graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    # A triangle's three corners are connected to each other by construction,
    # so any one of them names the component.
    per_element = labels[triangles[:, 0]]
    found = [np.nonzero(per_element == c)[0]
             for c in np.unique(per_element)]
    return sorted(found, key=len, reverse=True)


def submesh(mesh, elements):
    """Extract a subset of elements as a standalone Mesh.

    Unreferenced vertices are dropped and the connectivity remapped. Groups are
    carried over, restricted to the elements that survive and reindexed;
    a group with nothing left is omitted rather than kept empty.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        elements: element indices to keep. Order is preserved, so the caller
            controls the new element numbering.
    """
    elements = np.asarray(elements, dtype=int).ravel()
    tris = mesh.triangles[elements]
    used, remapped = np.unique(tris, return_inverse=True)

    # Position of each kept element in the new numbering, for rebasing groups.
    where = np.full(mesh.n_elements, -1, dtype=int)
    where[elements] = np.arange(len(elements))

    groups = {}
    for name, idx in mesh.groups.items():
        kept = where[np.asarray(idx, dtype=int)]
        kept = kept[kept >= 0]
        if len(kept):
            groups[name] = np.sort(kept)

    return Mesh(mesh.vertices[used], remapped.reshape(tris.shape), groups)


def triangle_keys(mesh, decimals=12):
    """Every triangle as an order-independent tuple of rounded coordinates.

    The only way to compare two meshes across an operation that renumbers
    things. mmgs and manifold3d both renumber vertices and triangles freely, so
    an index says nothing about identity and a coordinate set says everything.
    Sorting the three corners makes the key insensitive to winding and to which
    corner is written first.
    """
    rounded = np.round(np.asarray(mesh.vertices)[mesh.triangles], decimals)
    return [tuple(sorted(map(tuple, tri))) for tri in rounded]


def changed_elements(original, result, decimals=12):
    """Elements of ``result`` that are not verbatim triangles of ``original``.

    The dirty zone after a union or a remesh: everything the operation created
    or reshaped, and nothing it left alone. Feed it to :func:`dilate_elements`
    for a halo, then freeze the complement.

    Returns:
        ``(k,) int`` element indices into ``result``.
    """
    before = set(triangle_keys(original, decimals))
    return np.array([i for i, k in enumerate(triangle_keys(result, decimals))
                     if k not in before], dtype=int)


def unchanged_elements(original, result, decimals=12):
    """The complement of :func:`changed_elements` — what survived verbatim."""
    changed = set(changed_elements(original, result, decimals).tolist())
    return np.array([i for i in range(result.n_elements) if i not in changed],
                    dtype=int)


def dilate_elements(mesh, elements, radius, *, geodesic=True):
    """Grow an element set to everything within ``radius`` of it.

    Turns "the triangles the union touched" into "the region it is safe to let
    mmgs rework": a seam needs room either side of it to be remeshed into
    something well-shaped, and freezing right up to the cut leaves mmgs no
    slack.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        elements: seed element indices.
        radius: metres.
        geodesic: measure along the surface (default) rather than through
            space, so a halo around a seam on the temple does not swallow the
            pinna a few millimetres away across the air gap.

    Returns:
        ``(k,) int`` element indices, a superset of ``elements``.
    """
    elements = np.asarray(elements, dtype=int).ravel()
    if elements.size == 0:
        return elements

    source_nodes = np.unique(mesh.triangles[elements])
    if geodesic:
        d = geodesic_distance(mesh.vertices, mesh.triangles, source_nodes)
    else:
        from scipy.spatial import cKDTree              # lazy: scipy is an extra
        d = cKDTree(mesh.vertices[source_nodes]).query(mesh.vertices)[0]

    # A triangle joins the zone when any corner is within the radius: growing
    # by "all corners" would leave a one-element-wide unfrozen rim behind.
    return np.nonzero((d[mesh.triangles] <= radius).any(axis=1))[0]


def distance_to_mesh(points, mesh):
    """Exact point-to-surface distance for each point, (N,).

    Thin wrapper over the Eberly closest-point kernel already used to validate
    field points, so proximity here and point-dropping there cannot disagree.
    Vertex-to-vertex distance is *not* a substitute: it overestimates whenever
    the nearest feature is the middle of a face, which is the common case
    between two smooth bodies and would report a gap that is not there.
    """
    from bemkit.solvers.fieldpoints import min_distance_to_mesh    # lazy
    return min_distance_to_mesh(np.atleast_2d(points),
                                mesh.vertices, mesh.triangles)


def signed_distance_to_mesh(points, mesh):
    """Point-to-surface distance, negative inside the closed surface, (N,).

    Only meaningful for a watertight, outward-oriented ``mesh`` — check with
    ``quality.check_watertight`` first if that is not already known.
    """
    from bemkit.solvers.fieldpoints import points_inside_mesh      # lazy
    points = np.atleast_2d(points)
    d = distance_to_mesh(points, mesh)
    inside = points_inside_mesh(points, mesh.vertices, mesh.triangles)
    return np.where(inside, -d, d)


def min_clearance(mesh_a, mesh_b):
    """Smallest surface-to-surface distance between two bodies, in metres.

    Zero (or near it) means they touch or interpenetrate — this function does
    not distinguish the two; see ``quality.check_interpenetration``.

    Measured from ``mesh_a``'s vertices to ``mesh_b``'s faces *and* the other
    way round, because either body can have the offending feature in the middle
    of a face where the other has a vertex.
    """
    return float(min(distance_to_mesh(mesh_a.vertices, mesh_b).min(),
                     distance_to_mesh(mesh_b.vertices, mesh_a).min()))


def push_clear(device, head, gap, *, falloff=None, smoothing=3, max_iter=25,
               tol=1e-4):
    """Displace ``device`` outward until it clears ``head`` by ``gap``.

    The alternative to fusing two touching bodies: keep them as separate closed
    surfaces and guarantee an air gap wide enough that the BEM panels either
    side of it are not near-singular. The rule of thumb is ``gap`` of at least
    one local element size — below that, two panels see each other at less than
    their own width and conditioning degrades.

    **This distorts the device.** It is a deliberate approximation, not a fix:
    a temple pushed 2.5 mm off the pinna is no longer resting on it. Compare
    ``np.linalg.norm(out.vertices - device.vertices, axis=1).max()`` against
    the feature sizes you care about before trusting the result.

    Each pass displaces every device vertex along the head's local outward
    normal by however much it falls short of ``gap``, ramps that displacement
    to zero over ``falloff`` so the device is not creased at the edge of the
    contact patch, smooths it over the device's own connectivity, and
    re-measures. Displacement moves vertices, which changes what is nearest,
    hence the iteration.

    Args:
        device: the :class:`~bemkit.model.Mesh` to move. Should be welded.
        head: the :class:`~bemkit.model.Mesh` to clear. Must be watertight and
            outward-oriented — the sign of the distance depends on it.
        gap: target clearance in metres.
        falloff: distance in metres over which the displacement decays to zero.
            Defaults to ``3 * gap``. Larger is smoother and moves more of the
            device; smaller concentrates the distortion.
        smoothing: Laplacian passes applied to the displacement field each
            iteration. 0 disables smoothing.
        max_iter: give up after this many passes. Convergence is
            geometric, not linear — smoothing spreads each pass's
            displacement, so the last few percent of the gap take
            disproportionately many passes. Measured on the glasses,
            8 passes reached 3.36 mm of a 3.50 mm target.
        tol: convergence tolerance as a fraction of ``gap``. Exact convergence
            is unreachable in floating point — the pass that lands a vertex on
            the target leaves a residual shortfall of a few ulp — so a run that
            has plainly succeeded would otherwise iterate to ``max_iter`` and
            raise. The default is 0.01% of the gap.

    Returns:
        A new Mesh with the same connectivity and groups, vertices displaced.

    Note:
        The guarantee is at the device's **vertices**. A facet spanning a dip
        in the head can still come marginally closer than ``gap`` in its
        interior, so verify the assembled result with :func:`min_clearance`
        (which measures point-to-face) rather than assuming this succeeded.

    Raises:
        ValueError: ``gap`` is not positive.
        RuntimeError: the gap could not be reached in ``max_iter`` passes,
            naming the clearance actually achieved.
    """
    if gap <= 0:
        raise ValueError(f"push_clear: gap must be positive, got {gap}")
    falloff = 3.0 * gap if falloff is None else float(falloff)

    from scipy.spatial import cKDTree                  # lazy: scipy is an extra

    head_normals = vertex_normals(head.vertices, head.triangles)
    head_tree = cKDTree(head.vertices)

    # Neighbour lists for Laplacian smoothing, built once — connectivity does
    # not change, only coordinates.
    if smoothing:
        e = np.vstack([device.triangles[:, [0, 1]],
                       device.triangles[:, [1, 2]],
                       device.triangles[:, [2, 0]]])
        e = np.vstack([e, e[:, ::-1]])
        counts = np.bincount(e[:, 0], minlength=device.n_nodes)
        counts = np.maximum(counts, 1)

    atol = tol * gap
    verts = np.array(device.vertices, dtype=float, copy=True)
    for _ in range(max_iter):
        signed = signed_distance_to_mesh(verts, head)
        shortfall = gap - signed                       # >0 where too close
        if shortfall.max() <= atol:
            return Mesh(verts, device.triangles, device.groups)

        # Ramp to zero over `falloff` past the target, so vertices just outside
        # the contact patch move a little and the transition has no crease.
        magnitude = np.clip(shortfall, 0.0, None)
        taper = np.clip(1.0 - (signed - gap) / falloff, 0.0, 1.0)
        magnitude = magnitude * np.where(signed >= gap, taper, 1.0)

        if smoothing:
            for _ in range(smoothing):
                summed = np.bincount(e[:, 0], weights=magnitude[e[:, 1]],
                                     minlength=device.n_nodes)
                # Never smooth a vertex *below* what it needs, or the pass
                # undoes itself where the penetration is deepest.
                magnitude = np.maximum(magnitude, summed / counts)

        verts = verts + magnitude[:, None] * head_normals[head_tree.query(verts)[1]]

    achieved = signed_distance_to_mesh(verts, head).min()
    raise RuntimeError(
        f"push_clear: could not reach a {gap * 1e3:.2f} mm gap in {max_iter} "
        f"passes — best clearance {achieved * 1e3:.2f} mm. Raise max_iter, or "
        f"raise falloff so the displacement spreads over more of the device.")
