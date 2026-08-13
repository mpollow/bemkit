#!/usr/bin/env python3
"""How mesh/*.msh was generated from geometry/sphcap-50-20.step.

mesh/coarse.msh, standard.msh and fine.msh are already committed and have
solved results under runs/ -- running this again would regenerate meshes
that may differ slightly from the committed ones (gmsh gives no byte-level
determinism guarantee) and desync them from those results (caught by the
model.py fingerprint on the next `bem solve`, not silently). Recorded here
for provenance, as mesh_plate.py is for models/plate -- do not run this
against an existing mesh/ you want to keep.

    python models/sphcap/mesh_sphcap.py
"""

from pathlib import Path

from bemkit.geometry.generate import mesh_from_step
from bemkit.geometry.sizing import distance_field

HERE = Path(__file__).parent
STEP = HERE / "geometry" / "sphcap-50-20.step"
MESH = HERE / "mesh"

# mm -- same ramp as the pre-restructure config.toml this replaces.
DIST_MIN, DIST_MAX = 50.0, 100.0

# name -> (lc_min, lc_max), mm.
SIZING = {
    "standard": (5.0, 25.0),
    "fine": (2.5, 12.0),
    "coarse": (10.0, 50.0),
}


def main():
    MESH.mkdir(exist_ok=True)
    for name, (lc_min, lc_max) in SIZING.items():
        mesh_from_step(STEP, MESH / f"{name}.msh",
                        distance_field(lc_min, lc_max, DIST_MIN, DIST_MAX))


if __name__ == "__main__":
    main()
