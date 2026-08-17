#!/usr/bin/env python3
"""Generate the mesh/*.msh variants for models/sphcap_meshing.

Run once, by hand; the resulting meshes are committed so `bem check`/`solve`
work without gmsh installed. Re-running regenerates meshes that will differ
slightly from the committed ones (gmsh gives no byte-level determinism
guarantee) and desyncs them from runs/ -- caught by the model.py fingerprint
on the next `bem solve`, not silently.

    python models/sphcap_meshing/mesh_sphcap_meshing.py

The geometry is parametric (bemkit.geometry.mesh_sphere_cap), not a committed
STEP file: the whole point of the study is to sweep the discretisation of a
shape whose exact answer is known, so the shape itself has to be a parameter.
"""

import math
from pathlib import Path

import numpy as np

from bemkit.geometry import (
    distance_field, edge_field, geodesic_field, load_mesh, mesh_sphere_cap,
    uniform_field,
)

HERE = Path(__file__).parent
MESH = HERE / "mesh"

#: Sphere radius (mm) and cap half-angle (deg) -- 100 mm diameter, +/-30 deg.
#: Must stay in step with model.py's RADIUS_MM / ANGLE_DEG, which is what the
#: closed form is evaluated with.
RADIUS_MM = 50.0
ANGLE_DEG = 30.0

#: The band tops out here, so this is the wavelength every size is quoted
#: against. c = 340 m/s, matching model.py.
LAMBDA_MIN_MM = 340.0 / 16000.0 * 1e3        # 21.25 mm


def per_wavelength(n):
    """Element size (mm) giving `n` elements per wavelength at 16 kHz."""
    return LAMBDA_MIN_MM / n


#: name -> sizing builder. Every entry is a controlled comparison against
#: another entry; see README.md for which, and why.
SIZING = {
    # The convergence curve: one element size everywhere, 2 to 10 per
    # wavelength at the top of the band.
    "u2":  uniform_field(per_wavelength(2)),
    "u4":  uniform_field(per_wavelength(4)),
    "u6":  uniform_field(per_wavelength(6)),
    "u8":  uniform_field(per_wavelength(8)),
    "u10": uniform_field(per_wavelength(10)),

    # Graded towards the driven cap: fine where the source is, coarse on the
    # shadow side. cap_far vs u8 (same cap size) and vs u2 (same far size) is
    # the question "does grading towards the source buy anything at ka ~ 15".
    "cap_far": distance_field(per_wavelength(8), per_wavelength(2), 10.0, 70.0),
    "cap_mid": distance_field(per_wavelength(8), per_wavelength(4), 10.0, 70.0),

    # Same ramp as cap_far, measured over the surface instead of through
    # space. The sphere is convex, so this is a negative control: it must come
    # out ~the same as cap_far, or the models/plate result is suspect.
    "geodesic": geodesic_field(per_wavelength(8), per_wavelength(2), 10.0, 70.0),

    # A narrow band around the cap rim only, on an otherwise u4 mesh. The rim
    # is where the velocity BC jumps and where the faceted mesh under-states
    # the driven area -- the nearest thing this smooth geometry has to an
    # edge. vs u4 (same base size) and vs u6/u8 (comparable element count).
    "rim": edge_field(per_wavelength(16), per_wavelength(4), 2.0, 8.0),
}


def describe(path):
    """tris, mean element size, elements per wavelength at 16 kHz, cap area."""
    mesh = load_mesh(path)
    tri = mesh.vertices[mesh.triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 1e6
    h = np.sqrt(4 * area / np.sqrt(3))            # equivalent equilateral edge
    exact_cap = (2 * math.pi * RADIUS_MM ** 2
                 * (1 - math.cos(math.radians(ANGLE_DEG))))
    cap = area[mesh.groups["vibrating"]].sum()
    return dict(tris=len(mesh.triangles), h_mean=h.mean(), h_max=h.max(),
                per_wl=LAMBDA_MIN_MM / h.mean(), cap_ratio=cap / exact_cap)


def main():
    MESH.mkdir(exist_ok=True)
    for name, sizing in SIZING.items():
        mesh_sphere_cap(MESH / f"{name}.msh", sizing,
                        radius_mm=RADIUS_MM, angle_deg=ANGLE_DEG)

    print(f"\n{'variant':<10} {'tris':>7} {'h mean':>8} {'h max':>7} "
          f"{'el/wl@16k':>10} {'cap area':>10}")
    print("-" * 56)
    for name in SIZING:
        d = describe(MESH / f"{name}.msh")
        print(f"{name:<10} {d['tris']:>7d} {d['h_mean']:>7.2f}m "
              f"{d['h_max']:>6.2f}m {d['per_wl']:>10.1f} "
              f"{(d['cap_ratio'] - 1) * 100:>+9.2f}%")


if __name__ == "__main__":
    main()
