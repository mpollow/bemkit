"""Mesh diagnostics — the checks NumCalc does not make.

NumCalc validates almost nothing about a mesh. As the paper puts it, it "will
produce *some* results, and it is up to the user to check the validity". The
hints in arXiv:2312.00005 §5 name what actually goes wrong, and most of them are
cheap to test from the triangles alone. This module implements them; every
function is pure numpy and takes arrays, so it never needs the solver.

Each check returns a :class:`Finding`. Severity is deliberate:

* ``error`` — the result will be wrong, not merely inaccurate.
* ``warning`` — likely to degrade accuracy or stability.
* ``ok`` — checked and fine, reported so a clean bill of health is visible.

Hint coverage: 1 (orientation), 2 (elements per wavelength), 3 (source
clearance), 4 (element regularity), 6 (local size uniformity), plus the
watertightness that hints 1 and 7 both depend on.
"""

from collections import namedtuple

import numpy as np

#: One diagnostic result. ``detail`` carries the numbers behind the verdict.
Finding = namedtuple("Finding", "check level message detail")
Finding.__new__.__defaults__ = (None,)

LEVELS = ("ok", "warning", "error")


def _edges(triangles):
    """(3T, 2) directed edges of every triangle, in winding order."""
    return np.concatenate([triangles[:, [0, 1]],
                           triangles[:, [1, 2]],
                           triangles[:, [2, 0]]])


def check_watertight(vertices, triangles):
    """Every edge must be shared by exactly two triangles.

    A hole leaks the interior into the exterior and makes the BEM problem
    ill-posed; a non-manifold edge (three or more triangles) has no consistent
    normal. Both are errors, and both are invisible in a rendering.
    """
    e = np.sort(_edges(triangles), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    holes = int(np.sum(counts == 1))
    nonmanifold = int(np.sum(counts > 2))
    detail = {"boundary_edges": holes, "nonmanifold_edges": nonmanifold}
    if holes or nonmanifold:
        parts = []
        if holes:
            parts.append(f"{holes} boundary edge(s) — the surface has holes")
        if nonmanifold:
            parts.append(f"{nonmanifold} edge(s) shared by >2 triangles")
        return Finding("watertight", "error",
                       "; ".join(parts) + ". A BEM surface must be closed.",
                       detail)
    return Finding("watertight", "ok",
                   f"closed surface, {len(triangles)} triangles", detail)


def check_orientation(vertices, triangles):
    """Normals must be consistent *and* point away from the scatterer (Hint 1).

    Two separate questions, and NumCalc's own ``-check_normals`` only answers
    the first — a uniformly inverted mesh passes it. The second is settled here
    by the divergence theorem: for a closed, consistently wound surface the
    signed volume is positive exactly when the winding is outward.

    The paper calls a flipped normal the most common error that still produces
    plausible-looking output.
    """
    e = _edges(triangles)
    key = e[:, 0].astype(np.int64) * (len(vertices) + 1) + e[:, 1]
    rev = e[:, 1].astype(np.int64) * (len(vertices) + 1) + e[:, 0]
    # Consistent winding: every directed edge appears once, and its reverse
    # appears once (in the neighbouring triangle).
    consistent = (len(np.unique(key)) == len(key)
                  and np.isin(key, rev).all())
    if not consistent:
        return Finding("orientation", "error",
                       "triangle winding is inconsistent — neighbouring "
                       "elements disagree on which side is outside "
                       "(NumCalc's -check_normals reports the same)",
                       {"consistent": False})

    v = vertices[triangles]
    signed_volume = float(
        np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
    detail = {"consistent": True, "signed_volume": signed_volume}
    if signed_volume < 0:
        return Finding("orientation", "error",
                       f"normals point inward (signed volume "
                       f"{signed_volume:.4g} m³ < 0). Flip the winding — "
                       f"results will look plausible and be wrong.",
                       detail)
    return Finding("orientation", "ok",
                   f"consistent and outward (enclosed volume "
                   f"{signed_volume:.4g} m³)", detail)


def edge_lengths(vertices, triangles):
    """(T, 3) edge lengths of every triangle."""
    v = vertices[triangles]
    return np.linalg.norm(
        v[:, [1, 2, 0]] - v[:, [0, 1, 2]], axis=2)


def check_elements_per_wavelength(vertices, triangles, max_frequency,
                                  speed_of_sound=340.0, target=6.0):
    """At least ~6 elements per wavelength at the highest frequency (Hint 2).

    ``lc_max`` is *documented* as sizing for this, but nothing has ever checked
    the mesh that came out against the band that will be run.
    """
    lengths = edge_lengths(vertices, triangles)
    longest = float(lengths.max())
    wavelength = speed_of_sound / max_frequency
    per_wavelength = wavelength / longest
    detail = {"longest_edge": longest, "wavelength": wavelength,
              "elements_per_wavelength": per_wavelength,
              "max_frequency": max_frequency}
    msg = (f"{per_wavelength:.1f} elements per wavelength at "
           f"{max_frequency:.0f} Hz (longest edge {longest * 1e3:.1f} mm, "
           f"λ {wavelength * 1e3:.1f} mm)")
    if per_wavelength < target / 2:
        return Finding("elements_per_wavelength", "error",
                       f"{msg} — far below the ~{target:.0f} the paper asks "
                       f"for; the top of the band is not resolved", detail)
    if per_wavelength < target:
        return Finding("elements_per_wavelength", "warning",
                       f"{msg} — below the ~{target:.0f} rule of thumb", detail)
    return Finding("elements_per_wavelength", "ok", msg, detail)


def aspect_ratios(vertices, triangles):
    """Normalized aspect ratio per triangle: 1.0 is equilateral, higher is worse.

    ``longest_edge / (2 * inradius)``, scaled so an equilateral triangle scores
    exactly 1. The naive longest/shortest *edge* ratio is not good enough — a
    needle-flat triangle with edges 1.0, 0.5, 0.5 scores a harmless-looking 2
    on it while being degenerate, and that is precisely the sliver that ruins
    the integration.
    """
    lengths = edge_lengths(vertices, triangles)
    v = vertices[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    perimeter = lengths.sum(axis=1)
    inradius = 2.0 * area / np.maximum(perimeter, 1e-30)
    return lengths.max(axis=1) / np.maximum(2.0 * inradius, 1e-30) / np.sqrt(3)


def check_element_regularity(vertices, triangles, max_aspect=5.0):
    """Elements should be close to equilateral (Hint 4).

    A sliver integrates badly and degrades the whole system's conditioning.
    """
    aspect = aspect_ratios(vertices, triangles)
    bad = int(np.sum(aspect > max_aspect))
    detail = {"median": float(np.median(aspect)),
              "p99": float(np.percentile(aspect, 99)),
              "max": float(aspect.max()), "above_threshold": bad}
    msg = (f"aspect ratio median {detail['median']:.2f}, "
           f"99th pct {detail['p99']:.2f}, max {detail['max']:.2f}")
    if bad:
        return Finding("element_regularity", "warning",
                       f"{msg} — {bad} element(s) above {max_aspect:g}:1",
                       detail)
    return Finding("element_regularity", "ok", msg, detail)


def check_size_uniformity(vertices, triangles, max_ratio=4.0):
    """Neighbouring elements should be of similar size (Hint 6).

    Directly relevant to the graded meshes bemkit produces: an ``lc_min`` →
    ``lc_max`` ramp is exactly the case the hint warns about, and FMM stability
    depends on it.
    """
    v = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    e = np.sort(_edges(triangles), axis=1)
    owner = np.tile(np.arange(len(triangles)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = owner[:-1][same], owner[1:][same]
    if len(a) == 0:
        return Finding("size_uniformity", "ok", "no shared edges to compare",
                       {"pairs": 0})

    ratio = np.maximum(areas[a], areas[b]) / np.maximum(
        np.minimum(areas[a], areas[b]), 1e-30)
    bad = int(np.sum(ratio > max_ratio))
    detail = {"median": float(np.median(ratio)), "max": float(ratio.max()),
              "above_threshold": bad, "pairs": int(len(ratio))}
    msg = (f"neighbour area ratio median {detail['median']:.2f}, "
           f"max {detail['max']:.2f}")
    if bad:
        return Finding("size_uniformity", "warning",
                       f"{msg} — {bad} neighbouring pair(s) differ by more "
                       f"than {max_ratio:g}x", detail)
    return Finding("size_uniformity", "ok", msg, detail)


def check_source_clearance(vertices, triangles, sources, label="point source"):
    """A point source must clear the surface by ≥ 1 average edge length (Hint 3).

    The free-field Green's function is singular at zero distance, so a source
    sitting on the surface produces garbage rather than an error. bemkit already
    applies exactly this test to fieldpoints; sources never got it.
    """
    avg_edge = float(edge_lengths(vertices, triangles).mean())
    if len(sources) == 0:
        return Finding("source_clearance", "ok", "no point sources",
                       {"avg_edge": avg_edge})

    centroids = vertices[triangles].mean(axis=1)
    sources = np.asarray(sources, dtype=float).reshape(-1, 3)
    # Centroid distance is an upper bound on the true surface distance, so this
    # under-reports rather than crying wolf.
    d = np.linalg.norm(centroids[None, :, :] - sources[:, None, :],
                       axis=2).min(axis=1)
    detail = {"avg_edge": avg_edge, "min_distance": float(d.min()),
              "too_close": int(np.sum(d < avg_edge))}
    msg = (f"nearest {label} {d.min() * 1e3:.1f} mm from the surface "
           f"(average edge {avg_edge * 1e3:.1f} mm)")
    if d.min() < avg_edge:
        return Finding("source_clearance", "warning",
                       f"{msg} — closer than one average edge length; the "
                       f"Green's function is singular at zero distance",
                       detail)
    return Finding("source_clearance", "ok", msg, detail)


def check_shells(vertices, triangles, *, weld_tol=1e-9):
    """How many separate closed bodies the mesh actually contains.

    Exterior BEM is perfectly happy with several disjoint closed surfaces, so
    more than one shell is not an error — a head and a pair of glasses is two
    bodies by design. It is reported because the *count* is the cheapest way to
    tell a real boolean union from a mere join, and the two are otherwise
    indistinguishable: both are edge-manifold, both enclose a positive volume,
    and both pass every other check here. ``models/hutubs21_glassesearbuds``
    shipped a
    join believed to be a union for exactly that reason.

    Vertices are welded first, since an STL shares no corners and would
    otherwise report one shell per facet.
    """
    from bemkit.geometry.combine import shells
    from bemkit.model import Mesh

    mesh = Mesh(vertices, triangles)
    try:
        # shells() reaches for scipy at call time, and scipy is an extra.
        # A base install must still get a usable `bem check`.
        found = shells(mesh, tol=weld_tol)
    except ImportError:                               # pragma: no cover
        return Finding("shells", "ok",
                       "not checked — needs scipy (bemkit[meshing])", None)

    v = mesh.vertices[mesh.triangles]
    signed = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])) / 6.0
    volumes = [float(signed[s].sum()) for s in found]

    detail = {"n_shells": len(found),
              "elements": [int(len(s)) for s in found],
              "volumes_m3": volumes}

    if len(found) <= 1:
        return Finding("shells", "ok", "one connected closed body", detail)

    parts = ", ".join(f"{len(s)} tris / {vol * 1e3:.3g} L"
                      for s, vol in zip(found, volumes))
    return Finding(
        "shells", "ok",
        f"{len(found)} separate bodies ({parts}) — fine for an exterior "
        f"problem, but check this is a deliberate assembly and not a union "
        f"that silently failed to fuse",
        detail)


def check_interpenetration(vertices_a, triangles_a, vertices_b, triangles_b,
                           label="bodies"):
    """Whether body B pokes through body A's surface.

    Two overlapping closed surfaces are not a valid exterior scattering
    problem: the buried caps enclose cavities whose boundary data means
    nothing, and they contribute spurious interior resonances. The result still
    converges and still looks plausible, which is what makes this worth an
    explicit check rather than an assumption.
    """
    from bemkit.solvers.fieldpoints import (min_distance_to_mesh,
                                            points_inside_mesh)

    vertices_b = np.asarray(vertices_b, dtype=float)
    inside = points_inside_mesh(vertices_b, vertices_a, triangles_a)
    n = int(inside.sum())
    detail = {"n_inside": n, "n_vertices": len(vertices_b)}

    if not n:
        return Finding("interpenetration", "ok",
                       f"{label} do not overlap", detail)

    depth = min_distance_to_mesh(vertices_b[inside], vertices_a, triangles_a)
    detail["max_depth_m"] = float(depth.max())
    detail["median_depth_m"] = float(np.median(depth))
    return Finding(
        "interpenetration", "error",
        f"{n} of {len(vertices_b)} vertices of one body lie inside the other, "
        f"up to {depth.max() * 1e3:.2f} mm deep — the overlapping region "
        f"encloses a cavity with meaningless boundary conditions; fuse the "
        f"bodies or hold them apart",
        detail)


def check_clearance(vertices_a, triangles_a, vertices_b, triangles_b,
                    min_gap=None, gap_factor=1.0):
    """Whether two bodies come closer than their own elements are wide.

    Two panels separated by much less than their own width see each other
    almost singularly, and the system matrix pays for it. The rule of thumb is
    a gap of at least one local element size; below that, expect a rising
    iteration count and locally unreliable pressures.

    Args:
        min_gap: required clearance in metres. Defaults to ``gap_factor`` times
            the mean edge length of the two meshes, so the threshold scales
            with the discretisation rather than being an absolute guess.
        gap_factor: multiplier on the mean edge length when ``min_gap`` is None.
    """
    from bemkit.geometry.combine import distance_to_mesh
    from bemkit.model import Mesh

    mesh_a = Mesh(vertices_a, triangles_a)
    mesh_b = Mesh(vertices_b, triangles_b)

    if min_gap is None:
        mean_edge = np.mean(np.concatenate([
            edge_lengths(vertices_a, triangles_a),
            edge_lengths(vertices_b, triangles_b)]))
        min_gap = gap_factor * mean_edge

    d = np.concatenate([distance_to_mesh(mesh_a.vertices, mesh_b),
                        distance_to_mesh(mesh_b.vertices, mesh_a)])
    tight = int((d < min_gap).sum())
    detail = {"min_gap_m": float(min_gap), "closest_m": float(d.min()),
              "n_below": tight}

    if not tight:
        return Finding("clearance", "ok",
                       f"closest approach {d.min() * 1e3:.2f} mm, above the "
                       f"{min_gap * 1e3:.2f} mm threshold", detail)

    return Finding(
        "clearance", "warning",
        f"{tight} point(s) closer than {min_gap * 1e3:.2f} mm (closest "
        f"{d.min() * 1e3:.2f} mm) — panels nearer than their own width "
        f"condition the system badly; expect more solver iterations and "
        f"unreliable pressure in the gap",
        detail)


def check_mesh(vertices, triangles, *, max_frequency=None,
               speed_of_sound=340.0, sources=()):
    """Run every applicable check. Returns a list of :class:`Finding`."""
    findings = [check_watertight(vertices, triangles)]
    # Orientation is only meaningful on a closed surface — the signed-volume
    # argument assumes one, so don't report a bogus verdict on an open mesh.
    if findings[0].level == "ok":
        findings.append(check_orientation(vertices, triangles))
        findings.append(check_shells(vertices, triangles))
    findings.append(check_element_regularity(vertices, triangles))
    findings.append(check_size_uniformity(vertices, triangles))
    if max_frequency:
        findings.append(check_elements_per_wavelength(
            vertices, triangles, max_frequency, speed_of_sound))
    if len(sources):
        findings.append(check_source_clearance(vertices, triangles, sources))
    return findings


def worst_level(findings):
    """The most severe level present, for an exit code."""
    return max((f.level for f in findings), key=LEVELS.index, default="ok")
