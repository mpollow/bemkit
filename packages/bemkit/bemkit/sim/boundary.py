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

from collections import namedtuple

import numpy as np

from bemkit.config import _check_bc_entry
from bemkit.meshing.io import read_mesh_triangles
from pynumcalc.deck import _contiguous_ranges


#: One resolved boundary condition over an inclusive, 0-based element range.
#: ``kind`` is a NumCalc keyword (VELO/PRES/ADMI/IMPE) and the value is complex,
#: so a resolved entry says everything the deck needs.
BCRange = namedtuple("BCRange", "start end kind real imag curve")
BCRange.__new__.__defaults__ = (None,)   # curve name, ADMI only


def _entry_condition(val, default_velocity):
    """(kind, real, imag, curve) for one [BOUNDARY] entry.

    ``velocity``/``v`` stay the shorthand for a VELO condition; ``type``/
    ``value`` express the rest. Validated already by _check_bc_entry.
    """
    curve = val.get("curve")
    if "type" in val or "value" in val or curve is not None:
        kind = str(val.get("type", "velo")).upper()
        if curve is not None:
            # The curve carries the value; the ELEM line holds the constant it
            # scales, and NumCalc multiplies *each part by its own curve*. Both
            # constants must therefore be 1.0 — a 0.0 imaginary constant would
            # zero the imaginary curve out entirely, silently. (Mesh2HRTF's own
            # exporter writes 1.0/1.0 for the same reason.)
            value = val.get("value", [1.0, 1.0])
            if not isinstance(value, (list, tuple)):
                value = [float(value), float(value)]
            return kind, float(value[0]), float(value[1]), curve
        value = val.get("value", 0.0 if kind == "PRES" else 1.0)
        if isinstance(value, (list, tuple)):
            return kind, float(value[0]), float(value[1]), curve
        return kind, float(value), 0.0, curve
    return ("VELO",
            float(val.get("velocity", val.get("v", default_velocity))),
            0.0, None)


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


def resolve_boundary(boundary_cfg, variant, mesh_file=None, default_velocity=1.0):
    """Resolve a variant's [BOUNDARY] entries to element ranges.

    Parameters
    ----------
    boundary_cfg : dict or None
        The [BOUNDARY] table (``ModelConfig.boundary_cfg``): flat entries
        and/or per-variant subtables. Every entry is validated against the
        config grammar — malformed entries raise rather than silently
        resolving to "no BC" (which would export a rigid surface that
        solves to silence).
    variant : str
    mesh_file : Path or str or None
        Mesh the ranges are resolved against (only read if a geometric ``near=``
        entry is present; None raises for such entries).
    default_velocity : float
        Velocity for geometric entries without their own ``velocity``/``v``
        (``ModelConfig.default_velocity``).

    Returns
    -------
    dict
        ``{entry_key: [BCRange, ...]}`` with 0-based inclusive element ranges,
        each carrying its NumCalc condition kind and complex value. Empty dict
        if the variant has no BOUNDARY entries.

    Raises
    ------
    ValueError
        If an entry is malformed, or a ``near=`` anchor resolves to an element
        implausibly far from it (likely an anchor picked in the wrong units or
        coordinate frame).
    """
    bc = boundary_cfg or {}
    if not bc:
        return {}
    entries = _entries_for_variant(bc, variant)
    default_v = default_velocity

    centroids = None  # lazily loaded only if a geometric entry needs the mesh
    diag = None
    resolved = {}
    for key, val in entries.items():
        if isinstance(val, dict) and "near" not in val:
            # Another variant's subtable mixed into the flat entries — not ours.
            continue
        _check_bc_entry(key, val, f"BOUNDARY (variant '{variant}')")
        if isinstance(val, dict):
            if mesh_file is None:
                raise ValueError(
                    f"BOUNDARY '{key}' for variant '{variant}' uses a geometric "
                    f"'near=' anchor, which must be resolved against a mesh — "
                    f"call resolve_boundary with mesh_file (or export via "
                    f"export_variant).")
            if centroids is None:
                vertices, triangles = read_mesh_triangles(mesh_file)
                centroids = vertices[triangles].mean(axis=1)
                diag = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
            kind, re_, im_, curve = _entry_condition(val, default_v)
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
                ranges = [BCRange(s, e, kind, re_, im_, curve)
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
                    ranges.append(BCRange(idx, idx, kind, re_, im_, curve))
            resolved[key] = ranges
        else:
            # _check_bc_entry guarantees the shape: [s, e, v] or [[s, e, v], ...]
            # The literal form is velocity-only and stays that way; anything
            # else needs the geometric/typed form.
            rows = val if isinstance(val[0], list) else [val]
            resolved[key] = [BCRange(r[0], r[1], "VELO", float(r[2]), 0.0, None)
                             for r in rows]
    return resolved


def _is_point_list(near):
    """True if ``near`` is a list of points ([[x,y,z], ...]) vs a single point."""
    return len(near) > 0 and isinstance(near[0], (list, tuple))


def flatten(resolved):
    """Flatten a resolve_boundary() dict into a list of BCRange."""
    return [r for ranges in resolved.values() for r in ranges]
