"""Geometric boundary-condition resolution.

Boundary conditions can be given either as raw element indices (which are tied
to one specific mesh and go stale on every remesh) or as physical anchor points
resolved against the mesh at export time::

    [BOUNDARY.ear]
    left_ear = { near = [0.06, 0.0, 0.10] }              # nearest element
    loudspeaker = { near = [[x1, y1, z1], [x2, y2, z2]] } # one element per anchor
    cap = { near = [0.0, 0.0, 0.05], radius = 0.01 }     # all elements within 10 mm

Anchors and radii are in meters (like fieldpoint files). This is the library form of the
per-model ``find_bc`` scripts: element indices become *derived* data, so the
same ``[BOUNDARY.<target>]`` definition serves every mesh recipe of a target.

Resolution runs at export only; it never rewrites an already-exported bundle's
``NC.inp``. Literal ``[s, e, v]`` / ``[[s, e, v], ...]`` forms pass through
unchanged for backward compatibility.
"""

import numpy as np

from bemkit.meshing.io import read_mesh_triangles
from bemkit.sim.writers import _contiguous_ranges


def variant_target(variant):
    """Strip a ``__recipe`` / ``@profile`` suffix to get the BC-target name.

    ``ear__r05`` -> ``ear`` so one ``[BOUNDARY.ear]`` definition applies to all
    mesh recipes/profiles of the ``ear`` target.
    """
    cut = len(variant)
    for sep in ("__", "@"):
        i = variant.find(sep)
        if i != -1:
            cut = min(cut, i)
    return variant[:cut]


def _entries_for_variant(bc, variant):
    """Select the [BOUNDARY.<...>] entries that apply to a variant.

    A per-variant subsection is used exclusively when present, matched against
    the full variant name first, then the stripped target name; otherwise the
    global (flat) entries apply.
    """
    for key in (variant, variant_target(variant)):
        if key in bc and isinstance(bc[key], dict):
            return bc[key]
    return bc


def _nearest_element(centroids, point):
    """Return (index, distance) of the centroid nearest to ``point``."""
    d = np.linalg.norm(centroids - np.asarray(point, dtype=float), axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def resolve_boundary(config, variant, mesh_file):
    """Resolve a variant's [BOUNDARY] entries to element ranges.

    Parameters
    ----------
    config : ModelConfig or dict-like
        Must support ``get``. Reads ``config["BOUNDARY"]`` and, for geometric
        entries, ``config.get("velocity", 1.0)``.
    variant : str
    mesh_file : Path or str
        Mesh the ranges are resolved against (only read if a geometric ``near=``
        entry is present).

    Returns
    -------
    dict
        ``{entry_key: [(start, end, velocity), ...]}`` with 0-based inclusive
        element ranges. Empty dict if the variant has no BOUNDARY entries.

    Raises
    ------
    ValueError
        If a ``near=`` anchor resolves to an element implausibly far from it
        (likely an anchor picked in the wrong units or coordinate frame).
    """
    bc = config.get("BOUNDARY") or {}
    if not bc:
        return {}
    entries = _entries_for_variant(bc, variant)
    default_v = config.get("velocity", 1.0)

    centroids = None  # lazily loaded only if a geometric entry needs the mesh
    diag = None
    resolved = {}
    for key, val in entries.items():
        if isinstance(val, dict) and "near" in val:
            if centroids is None:
                vertices, triangles = read_mesh_triangles(mesh_file)
                centroids = vertices[triangles].mean(axis=1)
                diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
            v0 = val.get("velocity", val.get("v", default_v))
            near = val["near"]
            anchors = near if _is_point_list(near) else [near]
            radius = val.get("radius")
            if radius is not None:
                selected = set()
                for pt in anchors:
                    d = np.linalg.norm(centroids - np.asarray(pt, dtype=float),
                                       axis=1)
                    idx = np.nonzero(d <= float(radius))[0]
                    if len(idx) == 0:
                        raise ValueError(
                            f"BOUNDARY '{key}' anchor {list(pt)} for variant "
                            f"'{variant}' has no element centroid within radius "
                            f"{radius} m (nearest is {d.min():.4g} m away). "
                            f"Check the anchor/radius units and frame.")
                    selected.update(int(i) for i in idx)
                ranges = [(s, e, float(v0))
                          for s, e in _contiguous_ranges(selected)]
            else:
                ranges = []
                for pt in anchors:
                    idx, dist = _nearest_element(centroids, pt)
                    if diag and dist > 0.05 * diag:
                        raise ValueError(
                            f"BOUNDARY '{key}' anchor {list(pt)} for variant "
                            f"'{variant}' is {dist:.4g} m from its nearest element "
                            f"(>5% of the mesh extent {diag:.4g} m). Check the "
                            f"anchor's units/frame.")
                    ranges.append((idx, idx, float(v0)))
            resolved[key] = ranges
        elif isinstance(val, list) and len(val) >= 1:
            if isinstance(val[0], list):          # [[s, e, v], ...]
                resolved[key] = [(r[0], r[1], float(r[2])) for r in val if len(r) >= 3]
            elif len(val) >= 3:                   # [s, e, v]
                resolved[key] = [(val[0], val[1], float(val[2]))]
    return resolved


def _is_point_list(near):
    """True if ``near`` is a list of points ([[x,y,z], ...]) vs a single point."""
    return len(near) > 0 and isinstance(near[0], (list, tuple))


def flatten(resolved):
    """Flatten a resolve_boundary() dict into a list of (start, end, velocity)."""
    return [r for ranges in resolved.values() for r in ranges]
