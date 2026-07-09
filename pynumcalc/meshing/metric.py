#!/usr/bin/env python3
"""
Generate per-vertex isotropic metric (.sol) for mmgs remeshing.

Usage:
    python -m pynumcalc.meshing.metric INPUT.mesh OUTPUT.sol [options]

Options:
    --hmin FLOAT          Minimum element size (default: 0.001)
    --hmax FLOAT          Maximum element size (default: 0.010)
    --mode curvature|uniform  Sizing strategy (default: curvature)
    --refine X Y Z R H    Spherical refinement zone: center, radius, target size (repeatable)
    --npy FILE            Also save per-vertex sizes as .npy for inspection
"""

import argparse
import sys
import numpy as np
import meshio


def compute_curvature_sizes(vertices, triangles, hmin, hmax):
    """Estimate per-vertex mesh size from angular deviation of face normals.

    Flat regions get hmax, high-curvature regions get hmin.
    """
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

    h = hmax - (hmax - hmin) * curvature_norm
    return h, curv_min, curv_max


def apply_refinement_zones(h, vertices, refine_zones):
    """Apply spherical refinement overrides: take min of existing h and zone target."""
    counts = []
    for cx, cy, cz, radius, target in refine_zones:
        center = np.array([cx, cy, cz])
        dists = np.linalg.norm(vertices - center, axis=1)
        mask = dists <= radius
        h[mask] = np.minimum(h[mask], target)
        counts.append(mask.sum())
    return h, counts


def write_sol(sol_file, h):
    """Write Medit .sol file with per-vertex isotropic scalar metric."""
    n = len(h)
    with open(sol_file, "w") as f:
        f.write("MeshVersionFormatted\n 2\n\nDimension\n 3\n\n")
        f.write(f"SolAtVertices\n{n}\n1 1\n")
        for val in h:
            f.write(f" {val:.8g}\n")
        f.write("\nEnd\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate per-vertex isotropic metric (.sol) for mmgs remeshing."
    )
    parser.add_argument("mesh_file", help="Input .mesh file")
    parser.add_argument("sol_file", help="Output .sol metric file")
    parser.add_argument("--hmin", type=float, default=0.001, help="Minimum element size (default: 0.001)")
    parser.add_argument("--hmax", type=float, default=0.010, help="Maximum element size (default: 0.010)")
    parser.add_argument(
        "--mode",
        choices=["curvature", "uniform"],
        default="curvature",
        help="Sizing strategy (default: curvature)",
    )
    parser.add_argument(
        "--refine",
        nargs=5,
        metavar=("X", "Y", "Z", "R", "H"),
        type=float,
        action="append",
        default=[],
        help="Spherical refinement zone: center (x y z), radius R, target size H",
    )
    parser.add_argument("--npy", metavar="FILE", help="Also save per-vertex sizes as .npy")
    args = parser.parse_args(argv)

    print(f"Reading mesh: {args.mesh_file}")
    mesh = meshio.read(args.mesh_file)
    vertices = mesh.points  # (N, 3)

    # Extract triangles
    triangles = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            triangles = cell_block.data
            break
    if triangles is None:
        print("Error: no triangle cells found in mesh.", file=sys.stderr)
        sys.exit(1)

    n_verts = len(vertices)
    n_tris = len(triangles)
    print(f"  {n_verts} vertices, {n_tris} triangles")

    if args.mode == "curvature":
        print("Computing curvature-based sizing...")
        h, curv_min, curv_max = compute_curvature_sizes(vertices, triangles, args.hmin, args.hmax)
        print(f"  Curvature range: [{curv_min:.4f}, {curv_max:.4f}] rad")
    else:
        h = np.full(n_verts, args.hmax)
        print(f"Uniform sizing: h = {args.hmax}")

    if args.refine:
        print(f"Applying {len(args.refine)} refinement zone(s)...")
        h, counts = apply_refinement_zones(h, vertices, args.refine)
        for i, (zone, count) in enumerate(zip(args.refine, counts)):
            cx, cy, cz, radius, target = zone
            print(f"  Zone {i+1}: center=({cx},{cy},{cz}), R={radius}, h={target} → {count} vertices")

    print(f"Metric summary: h in [{h.min():.6g}, {h.max():.6g}], mean={h.mean():.6g}")

    print(f"Writing: {args.sol_file}")
    write_sol(args.sol_file, h)

    if args.npy:
        np.save(args.npy, h)
        print(f"Saved .npy: {args.npy}")

    print("Done.")


if __name__ == "__main__":
    main()
