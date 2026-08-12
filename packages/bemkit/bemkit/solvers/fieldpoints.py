"""Field-point validation and offset bookkeeping for the solver layer.

Ported from ``bemkit.sim.fieldpoints`` (the old sim layer this replaces): the
Möller-Trumbore / Eberly geometry kernels are unchanged, only the entry
points differ — they take a :class:`~bemkit.model.mesh.Mesh` rather than a
mesh-file path, since the model layer already holds vertices/triangles in
memory and has no file to re-read.
"""

import numpy as np

# ─── Geometry kernels (pure numpy) ────────────────────────────────────────


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


# ─── Public API ─────────────────────────────────────────────────────────


def validate(points, mesh, *, exterior=True, min_surface_distance=None,
             edge_length_factor=0.1):
    """Which of ``points`` are usable evaluation points for ``mesh``.

    A point is dropped when it is on the wrong side of the surface (inside
    for an exterior problem, outside for an interior one — NumCalc's own BEM
    formulation breaks down otherwise) or too close to it (the free-field
    Green's function is singular at zero distance).

    Args:
        points: (N, 3) evaluation points, metres.
        mesh: the Mesh the points are evaluated against.
        exterior: True (default) for a radiation/scattering problem outside
            the surface — matches ``Model.exterior``.
        min_surface_distance: minimum allowed distance to the surface,
            metres. If None, auto-computed as
            ``edge_length_factor * average mesh edge length``.
        edge_length_factor: see above.

    Returns:
        Boolean array (N,) — True where the point is usable.
    """
    points = np.asarray(points, dtype=float)
    vertices, triangles = mesh.vertices, mesh.triangles

    if min_surface_distance is None:
        edges = np.concatenate([
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 1]],
            vertices[triangles[:, 0]] - vertices[triangles[:, 2]],
        ])
        avg_edge = np.mean(np.linalg.norm(edges, axis=1))
        min_surface_distance = edge_length_factor * avg_edge

    inside = points_inside_mesh(points, vertices, triangles)
    valid = ~inside if exterior else inside

    dist = min_distance_to_mesh(points, vertices, triangles)
    valid &= dist >= min_surface_distance

    return valid


def combine(fields, mask=None):
    """Concatenate ``{name: (N, 3)}`` field-point sets into one array.

    Args:
        fields: ``{name: (N, 3)}`` arrays, in dict iteration order — that
            order is what the combined array (and NumCalc's EvalGrid) uses.
        mask: optional (M,) boolean keep-mask over the *concatenated* array
            (M = total points across every set, before filtering) — e.g.
            from calling :func:`validate` once on the concatenation. Rows
            where mask is False are dropped from both the returned array
            and the offsets.

    Returns:
        ``(points, offsets)``. ``points`` is ``(K, 3)`` (K = M when mask is
        None, else ``mask.sum()``). ``offsets[name]`` is an int array of row
        indices into ``points`` for that set, re-indexed to the *filtered*
        array — so it stays correct even when an earlier set in the dict
        lost points a later one didn't.
    """
    names = list(fields)
    arrays = [np.asarray(fields[name], dtype=float) for name in names]
    lengths = [len(a) for a in arrays]
    combined = np.vstack(arrays) if arrays and sum(lengths) else np.empty((0, 3))
    total = len(combined)

    if mask is None:
        keep_index = np.arange(total)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (total,):
            raise ValueError(
                f"mask must have shape ({total},) to match the concatenated "
                f"points, got {mask.shape}")
        keep_index = np.full(total, -1, dtype=int)
        keep_index[mask] = np.arange(int(mask.sum()))

    offsets = {}
    start = 0
    for name, n in zip(names, lengths):
        idx = keep_index[start:start + n]
        offsets[name] = idx[idx >= 0]
        start += n

    return (combined if mask is None else combined[mask]), offsets
