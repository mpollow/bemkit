"""Per-vertex isotropic metric (.sol) for mmgs curvature-adaptive remeshing.

``curvature_sizes`` estimates a target element size at every vertex from the
angular deviation of its incident face normals: flat regions get ``hmax``,
sharply curved regions get ``hmin``. ``write_sol`` serialises the result as a
Medit ``.sol`` file, the per-vertex isotropic metric ``mmgs`` reads to drive
its remeshing.
"""

import numpy as np


def curvature_sizes(mesh, hmin, hmax):
    """Estimate per-vertex mesh size from angular deviation of face normals.

    Flat regions get ``hmax``, high-curvature regions get ``hmin``.

    Args:
        mesh: a bemkit.model.Mesh.
        hmin: element size at the most curved vertex, in mesh units.
        hmax: element size on flat regions, in mesh units.

    Returns:
        (n_nodes,) array of target sizes, one per vertex, bounded by
        [hmin, hmax].
    """
    vertices, triangles = mesh.vertices, mesh.triangles

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0
    face_normals_raw = np.cross(e1, e2)  # (F, 3)
    face_areas = np.linalg.norm(face_normals_raw, axis=1) / 2.0  # (F,)
    face_normals = face_normals_raw / (face_areas[:, None] * 2.0 + 1e-15)  # unit normals

    # Area-weighted vertex normals
    n_verts = len(vertices)
    vertex_normals = np.zeros((n_verts, 3))
    vertex_area_sum = np.zeros(n_verts)

    for fi, (a, b, c) in enumerate(triangles):
        w = face_areas[fi]
        vertex_normals[a] += w * face_normals[fi]
        vertex_normals[b] += w * face_normals[fi]
        vertex_normals[c] += w * face_normals[fi]
        vertex_area_sum[a] += w
        vertex_area_sum[b] += w
        vertex_area_sum[c] += w

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(norms, 1e-15)

    # Per-vertex curvature: area-weighted angular deviation of incident faces
    curvature = np.zeros(n_verts)
    for fi, (a, b, c) in enumerate(triangles):
        fn = face_normals[fi]
        w = face_areas[fi]
        for vi in (a, b, c):
            cos_angle = np.clip(np.dot(fn, vertex_normals[vi]), -1.0, 1.0)
            curvature[vi] += w * np.arccos(np.abs(cos_angle))

    # Normalize by area sum
    curvature /= np.maximum(vertex_area_sum, 1e-15)

    curv_min, curv_max = curvature.min(), curvature.max()
    if curv_max > curv_min:
        curvature_norm = (curvature - curv_min) / (curv_max - curv_min)
    else:
        curvature_norm = np.zeros(n_verts)

    # Clip: the affine map lands exactly on hmin/hmax only up to float
    # rounding, and callers (mmgs) may reject a size that undershoots hmin.
    return np.clip(hmax - (hmax - hmin) * curvature_norm, hmin, hmax)


def write_sol(path, sizes):
    """Write an mmgs isotropic metric (.sol) — one size per vertex."""
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\n\nDimension 3\n\nSolAtVertices\n")
        f.write(f"{len(sizes)}\n1 1\n\n")
        f.writelines(f"{s:.10g}\n" for s in sizes)
        f.write("\nEnd\n")
