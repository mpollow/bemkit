#!/usr/bin/env python3
"""Compare euclidean vs geodesic element size on the front and back faces.

    python models/plate/compare_meshes.py

The membrane is on the front face (+z). Under euclidean sizing the back face
directly behind it is refined (5 mm straight-through); under geodesic it stays
at lc_max because the surface path runs around the plate edge. Run
mesh_plate.py first if mesh/*.msh does not exist yet.
"""
from pathlib import Path
import numpy as np
import meshio

MESH = Path(__file__).parent / "mesh"


def face_sizes(variant):
    m = meshio.read(MESH / f"{variant}.msh")
    tri, pts = m.cells_dict["triangle"], m.points
    c = pts[tri].mean(axis=1)
    a = np.linalg.norm(np.cross(pts[tri[:, 1]] - pts[tri[:, 0]],
                                 pts[tri[:, 2]] - pts[tri[:, 0]]), axis=1) / 2
    size = np.sqrt(4 * a / np.sqrt(3))
    zmax, zmin = c[:, 2].max(), c[:, 2].min()
    front = c[:, 2] > zmax - 1.0        # +z face (membrane side)
    back = c[:, 2] < zmin + 1.0         # -z face
    return size, front, back


def report():
    print(f"{'variant':<10} {'front mm':>10} {'back mm':>10} {'back/front':>11}")
    print("-" * 44)
    for v in ("euclidean", "geodesic"):
        size, front, back = face_sizes(v)
        f, b = size[front].mean(), size[back].mean()
        print(f"{v:<10} {f:>10.2f} {b:>10.2f} {b / f:>11.2f}x")
    print("\nGeodesic should leave the back face near lc_max while euclidean "
          "refines it; expect geodesic back/front >> euclidean back/front.")


if __name__ == "__main__":
    report()
