"""Fieldpoint validation, loading, and offset management for BEM simulations."""

import json
import numpy as np
from pathlib import Path


# ─── Geometry kernels (pure numpy) ───────────────────────────────────────────

def points_inside_mesh(points, vertices, triangles):
    """Ray-cast each point in multiple directions. Majority vote = inside.

    Uses 3 orthogonal ray directions (+x, +y, +z) to handle open surfaces
    and edge cases: a point is inside if at least 2/3 rays report an odd
    intersection count. Chunked Möller-Trumbore, vectorized over triangle
    batches instead of looping per-triangle.

    Args:
        points: (N, 3) query points
        vertices: (M, 3) mesh vertices
        triangles: (T, 3) triangle vertex indices (0-indexed)

    Returns:
        Boolean array (N,) — True where point is inside the mesh
    """
    eps = 1e-12
    N = len(points)
    T = len(triangles)

    v0s = vertices[triangles[:, 0]]  # (T, 3)
    v1s = vertices[triangles[:, 1]]
    v2s = vertices[triangles[:, 2]]
    e1s = v1s - v0s  # (T, 3)
    e2s = v2s - v0s  # (T, 3)

    test_directions = [
        np.array([0.0, 0.0, 1.0]),  # +z
        np.array([1.0, 0.0, 0.0]),  # +x
        np.array([0.0, 1.0, 0.0]),  # +y
    ]

    inside_votes = np.zeros(N, dtype=int)

    # Chunk size: target ~50 MB for the (N, C, 3) s array (N*C*24 bytes)
    C = max(16, min(512, 50_000_000 // max(1, N * 24)))

    for direction in test_directions:
        # Per-direction quantities precomputed over all T triangles
        h_all = np.cross(direction, e2s)           # (T, 3)
        a_all = np.einsum('ti,ti->t', e1s, h_all)  # (T,)
        parallel_all = np.abs(a_all) < eps
        f_all = np.where(parallel_all, 1.0, 1.0 / np.where(parallel_all, 1.0, a_all))  # (T,)

        counts = np.zeros(N, dtype=int)

        for start in range(0, T, C):
            end = min(start + C, T)
            h_c   = h_all[start:end]          # (Cc, 3)
            f_c   = f_all[start:end]          # (Cc,)
            par_c = parallel_all[start:end]   # (Cc,)
            v0_c  = v0s[start:end]            # (Cc, 3)
            e1_c  = e1s[start:end]            # (Cc, 3)
            e2_c  = e2s[start:end]            # (Cc, 3)

            s = points[:, None, :] - v0_c[None, :, :]  # (N, Cc, 3)

            u = f_c[None, :] * (s * h_c[None]).sum(-1)  # (N, Cc)
            valid = (~par_c[None, :]) & (u >= 0.0) & (u <= 1.0)

            q = np.cross(s, e1_c[None])  # (N, Cc, 3)

            v = f_c[None, :] * (q * direction[None, None]).sum(-1)  # (N, Cc)
            valid &= (v >= 0.0) & (u + v <= 1.0)

            t_vals = f_c[None, :] * (q * e2_c[None]).sum(-1)  # (N, Cc)
            valid &= t_vals > eps

            counts += valid.sum(axis=1)

        inside_votes += counts % 2

    return inside_votes >= 2


def min_distance_to_mesh(points, vertices, triangles):
    """Compute minimum distance from each point to the mesh surface.

    Chunked Eberly closest-point algorithm, vectorized over triangle
    batches using BLAS matmul for the d/e terms.

    Args:
        points: (N, 3) query points
        vertices: (M, 3) mesh vertices
        triangles: (T, 3) triangle vertex indices (0-indexed)

    Returns:
        (N,) minimum distances
    """
    N = len(points)
    T = len(triangles)

    v0s = vertices[triangles[:, 0]]  # (T, 3)
    v1s = vertices[triangles[:, 1]]
    v2s = vertices[triangles[:, 2]]
    e0s = v1s - v0s  # (T, 3)
    e1s = v2s - v0s  # (T, 3)

    # Precompute per-triangle scalars
    a_all   = np.einsum('ti,ti->t', e0s, e0s)       # (T,)
    b_all   = np.einsum('ti,ti->t', e0s, e1s)       # (T,)
    c_all   = np.einsum('ti,ti->t', e1s, e1s)       # (T,)
    det_all = a_all * c_all - b_all * b_all          # (T,)

    # For BLAS trick: d = dot(v0-p, e0) = dot(v0, e0) - dot(p, e0)
    v0_dot_e0 = np.einsum('ti,ti->t', v0s, e0s)     # (T,)
    v0_dot_e1 = np.einsum('ti,ti->t', v0s, e1s)     # (T,)

    # Chunk size: target ~50 MB for (N, C, 3) arrays
    C = max(16, min(512, 50_000_000 // max(1, N * 24)))

    min_dist = np.full(N, np.inf)

    for start in range(0, T, C):
        end = min(start + C, T)
        sl = slice(start, end)

        a_c   = a_all[sl];   b_c = b_all[sl];   c_c = c_all[sl]
        det_c = det_all[sl]
        v0_c  = v0s[sl];     e0_c = e0s[sl];    e1_c = e1s[sl]

        # d, e via BLAS matmul — avoids building (N, Cc, 3) diff
        d = v0_dot_e0[None, sl] - (points @ e0_c.T)  # (N, Cc)
        e = v0_dot_e1[None, sl] - (points @ e1_c.T)  # (N, Cc)

        s_vals = b_c[None] * e - c_c[None] * d   # (N, Cc)
        t_vals = b_c[None] * d - a_c[None] * e   # (N, Cc)

        # Safe divisors (broadcast from (Cc,) to (N, Cc))
        safe_a   = np.where(a_c   > 1e-30, a_c,   1.0)
        safe_c   = np.where(c_c   > 1e-30, c_c,   1.0)
        safe_det = np.where(det_c > 1e-30, det_c, 1.0)
        inv_det  = np.where(det_c > 1e-30, 1.0 / safe_det, 0.0)
        denom_h  = a_c - 2 * b_c + c_c
        safe_hyp = np.where(np.abs(denom_h) > 1e-30, denom_h, 1.0)

        # Region masks (N, Cc)
        inner = (s_vals + t_vals) <= det_c[None]
        s_nn  = s_vals >= 0
        t_nn  = t_vals >= 0

        m0  = inner &  s_nn &  t_nn
        m3  = inner & ~s_nn &  t_nn
        m5  = inner &  s_nn & ~t_nn
        m4a = inner & ~s_nn & ~t_nn & (d < 0)
        m4b = inner & ~s_nn & ~t_nn & (d >= 0)
        m1  = ~inner &  s_nn &  t_nn
        m2a = ~inner & ~s_nn & ((c_c[None] + e) > (b_c[None] + d))
        m2b = ~inner & ~s_nn & ((c_c[None] + e) <= (b_c[None] + d))
        m6a = ~inner &  s_nn & ~t_nn & ((a_c[None] + d) > (b_c[None] + e))
        m6b = ~inner &  s_nn & ~t_nn & ((a_c[None] + d) <= (b_c[None] + e))

        # Shared intermediate: s1 used by m1 and m2a, t6a used by m6a
        s1  = np.clip(((c_c[None] + e) - (b_c[None] + d)) / safe_hyp[None], 0.0, 1.0)
        t6a = np.clip(((a_c[None] + d) - (b_c[None] + e)) / safe_hyp[None], 0.0, 1.0)

        clip_s = np.clip(-d / safe_a[None], 0.0, 1.0)
        clip_t = np.clip(-e / safe_c[None], 0.0, 1.0)

        s_out = (
            m0  * (s_vals * inv_det[None]) +
            m5  * clip_s +
            m4a * clip_s +
            m1  * s1 +
            m2a * s1 +
            m6a * (1.0 - t6a) +
            m6b * clip_s
        )
        t_out = (
            m0  * (t_vals * inv_det[None]) +
            m3  * clip_t +
            m4b * clip_t +
            m1  * (1.0 - s1) +
            m2a * (1.0 - s1) +
            m2b * clip_t +
            m6a * t6a
        )

        # closest = v0 + s_out*e0 + t_out*e1  → (N, Cc, 3)
        closest = v0_c[None] + s_out[:, :, None] * e0_c[None] + t_out[:, :, None] * e1_c[None]
        diff_c  = points[:, None, :] - closest              # (N, Cc, 3)
        dist_c  = np.sqrt(np.einsum('nci,nci->nc', diff_c, diff_c))  # (N, Cc)

        min_dist = np.minimum(min_dist, dist_c.min(axis=1))

    return min_dist


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_points(points, mesh_file, is_internal=0, min_surface_distance=None, edge_length_factor=0.1):
    """Check which points are valid (outside mesh for external, inside for internal).

    Uses ray-casting to determine inside/outside and distance checking for
    points too close to the surface.

    Args:
        points: (N, 3) evaluation points in meters
        mesh_file: Path to mesh file (.msh, .stl, .ply, etc.)
        is_internal: 0 for external problem, 1 for internal
        min_surface_distance: Minimum allowed distance to surface in meters.
            If None, auto-computed as edge_length_factor × average mesh edge length.

    Returns:
        Boolean mask (N,) — True where point is valid
    """
    from bemkit.meshing.io import read_mesh_triangles
    vertices, triangles = read_mesh_triangles(mesh_file)

    # Auto-compute min_surface_distance from average edge length
    if min_surface_distance is None:
        edges = np.concatenate([
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 1]],
            vertices[triangles[:, 0]] - vertices[triangles[:, 2]],
        ])
        avg_edge = np.mean(np.linalg.norm(edges, axis=1))
        min_surface_distance = edge_length_factor * avg_edge
        print(f"  Auto min_surface_distance: {min_surface_distance*1000:.2f} mm "
              f"({edge_length_factor}x avg edge length {avg_edge*1000:.2f} mm)")

    inside = points_inside_mesh(points, vertices, triangles)

    if is_internal == 0:
        # External problem: points should be outside
        valid = ~inside
    else:
        # Internal problem: points should be inside
        valid = inside

    # Check minimum distance to surface
    dist = min_distance_to_mesh(points, vertices, triangles)
    too_close = dist < min_surface_distance
    valid &= ~too_close

    n_inside = np.sum(inside) if is_internal == 0 else np.sum(~inside)
    n_close = np.sum(too_close & (valid | too_close))  # those that would fail distance check
    if n_inside > 0 or np.sum(too_close) > 0:
        print(f"  Validation: {np.sum(~valid)} points dropped "
              f"({n_inside} wrong side, {np.sum(too_close)} too close to surface)")

    return valid


def _as_points(array, name):
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"fieldpoint set '{name}': expected an (N, 3) array, "
            f"got shape {array.shape}")
    return array


def load_fieldpoint_sets(fieldpoints_config, model_dir):
    """Load all fieldpoint sets from the [fieldpoints] config table.

    Each named subtable is one set, defined either declaratively
    (``type = "onaxis" | "polar" | "plane"`` plus generator kwargs — points
    are generated in-memory via make_fields_<type>) or from a file
    (``file = "path.npy"``, relative to model_dir). When no subtables are
    given, all ``fieldpoints/*.npy`` in the model directory are loaded, one
    set per file (a configured ``exec`` script, run by the pipeline before
    loading, typically writes them).

    Args:
        fieldpoints_config: [fieldpoints] table from config.toml (dict; the
            reserved ``exec`` key is ignored here — the pipeline runs it)
        model_dir: Model directory (paths resolved relative to this)

    Returns:
        Dict of {name: ndarray} with shape (N, 3) per set
    """
    from bemkit.sim import fields

    model_dir = Path(model_dir)
    specs = {name: spec for name, spec in (fieldpoints_config or {}).items()
             if name != "exec"}
    sets = {}
    for name, spec in specs.items():
        if "type" in spec:
            fn = getattr(fields, f"make_fields_{spec['type']}")
            kwargs = {k: v for k, v in spec.items() if k != "type"}
            try:
                points = fn(**kwargs)
            except (TypeError, ValueError) as e:
                raise ValueError(f"[fieldpoints.{name}]: {e}") from e
            print(f"  Generated '{name}': {len(points)} points "
                  f"({spec['type']})")
        else:
            path = Path(spec["file"])
            if not path.is_absolute():
                path = model_dir / path
            points = _as_points(np.load(path), name)
            print(f"  Loaded '{name}': {len(points)} points from {spec['file']}")
        sets[name] = points
    if not specs:
        fp_dir = model_dir / "fieldpoints"
        for npy_file in (sorted(fp_dir.glob("*.npy")) if fp_dir.is_dir() else []):
            sets[npy_file.stem] = _as_points(np.load(npy_file), npy_file.stem)
            print(f"  Loaded '{npy_file.stem}': {len(sets[npy_file.stem])} "
                  f"points from {npy_file}")
    if not sets:
        raise ValueError(
            f"No fieldpoint sets: define [fieldpoints.<name>] tables in "
            f"config.toml or put .npy files in {model_dir / 'fieldpoints'}")
    return sets


def combine_and_track(sets_dict, valid_masks=None):
    """Concatenate point arrays and track offsets.

    Args:
        sets_dict: Dict of {name: ndarray} point arrays
        valid_masks: Optional dict of {name: boolean_mask}. If provided,
            only valid points are included in the combined array.

    Returns:
        Tuple of (combined_array, offsets_dict) where offsets_dict has format:
        {
            "sets": {name: {"start": int, "end": int, "n_original": int, "n_valid": int}},
            "dropped": {name: [indices]}  # only for sets with dropped points
        }
    """
    combined = []
    offsets = {"sets": {}, "dropped": {}}
    current_idx = 0

    for name, points in sets_dict.items():
        n_original = len(points)

        if valid_masks and name in valid_masks:
            mask = valid_masks[name]
            valid_points = points[mask]
            dropped_indices = np.where(~mask)[0].tolist()
            if dropped_indices:
                offsets["dropped"][name] = dropped_indices
        else:
            valid_points = points
            dropped_indices = []

        n_valid = len(valid_points)
        offsets["sets"][name] = {
            "start": current_idx,
            "end": current_idx + n_valid,
            "n_original": n_original,
            "n_valid": n_valid,
        }

        if n_valid > 0:
            combined.append(valid_points)
        current_idx += n_valid

    combined_array = np.vstack(combined) if combined else np.empty((0, 3))
    return combined_array, offsets


def split_by_offsets(combined_array, offsets):
    """Split combined result array back into per-set arrays.

    If points were dropped during validation, reconstructs full arrays
    with NaN at invalid positions.

    Args:
        combined_array: (N_valid, ...) array of combined results
        offsets: Dict from combine_and_track() with "sets" and "dropped" keys

    Returns:
        Dict of {name: ndarray} with shape (n_original, ...) per set.
        Dropped positions filled with NaN.
    """
    result = {}
    dropped = offsets.get("dropped", {})

    for name, info in offsets["sets"].items():
        start = info["start"]
        end = info["end"]
        n_original = info["n_original"]
        n_valid = info["n_valid"]

        valid_data = combined_array[start:end]

        if name in dropped and dropped[name]:
            # Reconstruct full array with NaN for dropped points
            shape = (n_original,) + combined_array.shape[1:]
            full = np.full(shape, np.nan, dtype=combined_array.dtype)
            mask = np.ones(n_original, dtype=bool)
            mask[dropped[name]] = False
            full[mask] = valid_data
            result[name] = full
        else:
            result[name] = valid_data

    return result


def save_offsets(offsets, filepath):
    """Save eval_offsets.json."""
    with open(filepath, 'w') as f:
        json.dump(offsets, f, indent=2)


def load_offsets(filepath):
    """Load eval_offsets.json."""
    with open(filepath, 'r') as f:
        return json.load(f)
