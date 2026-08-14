"""Per-vertex target element size — the terms, and how to combine them.

An acoustic mesh has to satisfy several unrelated demands at once. It must
resolve the shortest wavelength in the band. It must resolve geometric detail,
which has nothing to do with frequency. It must be fine where two bodies nearly
touch, and it may as well be coarse on the back of the head where the field is
smooth. Writing that as one closed-form rule produces a function with eight
arguments that nobody can reason about; ``models/hutubs21_glassesearbuds``
instead wrote
it as thirteen hand-placed spheres with hardcoded radii.

So each demand is a separate function of the same shape --- ``(mesh, ...) ->
(N,) sizes in metres`` --- and :func:`finest` takes the elementwise minimum.
A size field is then an expression::

    sizes = finest(
        wavelength(mesh, f_max=16e3, c=340.0),              # the band
        curvature(mesh, hmin=2e-3, hmax=7e-3),              # the geometry
        near_elements(mesh, ears, 2e-3, 7e-3, 0, 30e-3),    # the region of interest
        near_mesh(mesh, glasses, 2e-3, 7e-3, 0, 10e-3),     # the junction
    )

which reads as the list of reasons the mesh is the way it is. The result feeds
``mmg.write_sol`` and drives ``mmgs``; it is deliberately independent of GMSH's
background-field machinery in ``sizing.py``, which solves the same problem for
the mesh-generation path where no mesh exists yet to hang per-vertex data on.

The ramp shape is shared with that path on purpose: every distance-based term
here maps distance to size through ``geodesic.size_from_distance``, the same
linear-with-clamps profile GMSH's ``Threshold`` field applies, so a study that
switches between generating and remeshing does not silently change grading.

numpy only, with scipy reached lazily by the distance terms.
"""

import numpy as np

from bemkit.geometry.geodesic import geodesic_distance, size_from_distance
from bemkit.geometry.normals import face_normals, vertex_areas, vertex_normals


def finest(*fields):
    """Elementwise minimum of several size fields.

    Combining by minimum is what makes the terms independent: adding a demand
    can only refine the mesh, never coarsen it, so terms compose in any order
    and no term has to know about the others.

    Args:
        *fields: ``(N,)`` arrays, all the same length.

    Returns:
        ``(N,)`` array of the smallest size requested at each vertex.

    Raises:
        ValueError: no fields, or the lengths disagree — which in practice
            means two of them were computed against different meshes, and the
            result would be silently meaningless.
    """
    if not fields:
        raise ValueError("finest: at least one size field is required")

    arrays = [np.asarray(f, dtype=float) for f in fields]
    lengths = {a.shape for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"finest: size fields must all have the same shape, got "
            f"{sorted(str(s) for s in lengths)} — they were probably computed "
            f"against different meshes")

    return np.minimum.reduce(arrays)


def uniform(mesh, h):
    """Constant size ``h`` everywhere, ``(N,)``.

    Mostly useful as a ceiling under :func:`finest`, so no other term can
    coarsen past a size you are willing to accept.
    """
    if h <= 0:
        raise ValueError(f"uniform: h must be positive, got {h}")
    return np.full(mesh.n_nodes, float(h))


def wavelength(mesh, f_max, c=343.0, per_wavelength=6.0):
    """The size the top of the band demands, ``(N,)`` (constant).

    ``h = c / (per_wavelength * f_max)``. Six elements per wavelength is what
    ``quality.check_elements_per_wavelength`` tests against and what
    arXiv:2312.00005 asks for; passing ``per_wavelength=3`` is a deliberate,
    documentable economy, not a default.

    Deriving the coarse size this way rather than typing a millimetre figure is
    the point — it keeps the mesh and the frequency band from drifting apart
    when one of them changes.
    """
    if f_max <= 0:
        raise ValueError(f"wavelength: f_max must be positive, got {f_max}")
    if per_wavelength <= 0:
        raise ValueError(
            f"wavelength: per_wavelength must be positive, got {per_wavelength}")
    return np.full(mesh.n_nodes, c / (per_wavelength * f_max))


def curvature(mesh, hmin, hmax):
    """Size from local curvature, ``(N,)`` — flat gets ``hmax``, creased gets
    ``hmin``.

    Curvature at a vertex is the area-weighted mean angle between its incident
    face normals and its own (area-weighted) normal, then rescaled so the
    flattest vertex on the mesh lands at ``hmax`` and the sharpest at ``hmin``.

    That rescaling is **relative to this mesh**: the same pinna sized alongside
    a torso and sized on its own gets different numbers, because the extremes
    it is normalised against differ. It is a shape-following heuristic, not an
    absolute measure, so pin the sizes you actually care about with
    :func:`sphere_zone` or a distance term rather than relying on it alone.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        hmin: size at the most curved vertex, metres.
        hmax: size on the flattest, metres.
    """
    if not 0 < hmin <= hmax:
        raise ValueError(
            f"curvature: need 0 < hmin <= hmax, got hmin={hmin}, hmax={hmax}")

    vertices, triangles = mesh.vertices, mesh.triangles
    fn, fa = face_normals(vertices, triangles)
    vn = vertex_normals(vertices, triangles)

    # Angle between each face's normal and each of its three corners' normals.
    # abs() before arccos folds the angle into [0, pi/2]: a normal that has
    # flipped relative to its neighbours is as "curved" as one at 90 degrees,
    # not more so, which keeps an inconsistently wound facet from dominating.
    cos_angle = np.einsum("fi,fci->fc", fn, vn[triangles])
    contrib = fa[:, None] * np.arccos(np.clip(np.abs(cos_angle), 0.0, 1.0))

    curv = np.bincount(triangles.ravel(), weights=contrib.ravel(),
                       minlength=mesh.n_nodes)
    curv /= np.maximum(vertex_areas(vertices, triangles), 1e-15)

    lo, hi = curv.min(), curv.max()
    normalised = (curv - lo) / (hi - lo) if hi > lo else np.zeros_like(curv)

    # Clip: the affine map lands exactly on hmin/hmax only up to float
    # rounding, and mmgs rejects a size that undershoots the hmin it was given.
    return np.clip(hmax - (hmax - hmin) * normalised, hmin, hmax)


def sphere_zone(mesh, center, radius, h, *, background=np.inf):
    """Clamp size to ``h`` inside a sphere, ``(N,)``.

    The blunt instrument, for when you know exactly where the detail is: an
    ear, a hinge, a nose bridge. Outside the sphere the field is ``background``
    (infinite by default, so under :func:`finest` the zone only ever refines
    and never coarsens anything else).

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        center: sphere centre, metres.
        radius: sphere radius, metres.
        h: size inside, metres.
        background: size outside. Leave at ``inf`` when composing.

    Raises:
        ValueError: the sphere contains no vertices — which almost always means
            millimetres were passed for metres, or the wrong coordinate frame.
    """
    d = np.linalg.norm(mesh.vertices - np.asarray(center, dtype=float), axis=1)
    inside = d <= radius
    if not inside.any():
        raise ValueError(
            f"sphere_zone: no vertices within {radius} m of {tuple(center)} — "
            f"nearest is {d.min():.4g} m away. Check units (this API is "
            f"metres) and the coordinate frame.")
    return np.where(inside, float(h), background)


def near_points(mesh, points, h_near, h_far, d_min, d_max):
    """Size ramped by straight-line distance to a set of points, ``(N,)``.

    ``h_near`` up to ``d_min``, linear to ``h_far`` at ``d_max``, clamped both
    ends. Use for an anchor that is not part of the mesh — a source position,
    an ear canal entrance — where there is no group to measure from.
    """
    from scipy.spatial import cKDTree                  # lazy: scipy is an extra

    points = np.atleast_2d(np.asarray(points, dtype=float))
    d = cKDTree(points).query(mesh.vertices)[0]
    return size_from_distance(d, h_near, h_far, d_min, d_max)


def near_elements(mesh, elements, h_near, h_far, d_min, d_max, *,
                  geodesic=True):
    """Size ramped by distance from a region of the mesh's own surface, ``(N,)``.

    Args:
        mesh: a :class:`~bemkit.model.Mesh`.
        elements: element indices of the region to measure from — typically
            ``mesh.groups["ear_left"]``.
        h_near, h_far: size at/below ``d_min`` and at/above ``d_max``, metres.
        d_min, d_max: ramp bounds, metres.
        geodesic: measure *along* the surface (the default) rather than through
            space. This matters most for exactly the geometry we care about: a
            point on the back of the pinna is a couple of millimetres from the
            concha through the air but a long way around the surface, and a
            straight-line ramp refines the whole outer ear for no gain. Needs
            the mesh to be welded, or every facet is its own island.

    Raises:
        ValueError: ``elements`` is empty — a ramp measured from nothing would
            silently return ``h_far`` everywhere.
    """
    elements = np.asarray(elements, dtype=int).ravel()
    if elements.size == 0:
        raise ValueError("near_elements: empty element set to measure from")

    source_nodes = np.unique(mesh.triangles[elements])

    if geodesic:
        d = geodesic_distance(mesh.vertices, mesh.triangles, source_nodes)
    else:
        from scipy.spatial import cKDTree              # lazy: scipy is an extra
        d = cKDTree(mesh.vertices[source_nodes]).query(mesh.vertices)[0]

    return size_from_distance(d, h_near, h_far, d_min, d_max)


def near_mesh(mesh, other, h_near, h_far, d_min, d_max):
    """Size ramped by distance to *another body's* surface, ``(N,)``.

    The junction term: refine the head where the glasses come close to it, and
    leave it alone elsewhere. Distance is point-to-face, so a coarse device
    does not fool it into thinking there is room.

    Applying this to both bodies with the same arguments is usually what you
    want — the fine region has to match on both sides of the gap, or the finer
    body ends up facing panels several times its own size.
    """
    from bemkit.geometry.combine import distance_to_mesh    # lazy: avoids cycle

    d = distance_to_mesh(mesh.vertices, other)
    return size_from_distance(d, h_near, h_far, d_min, d_max)
