#!/usr/bin/env python3
"""Generate the mesh/*.msh variants for models/plate from geometry/plate.step.

Run once, by hand -- there is no `bem mesh` verb any more (a study is
Python; see CLAUDE.md). Committing the resulting mesh/*.msh keeps `bem
models/plate check`/`solve` fast and reproducible without needing gmsh
installed just to load the study.

    python models/plate/mesh_plate.py
"""

from pathlib import Path

from bemkit.geometry.generate import mesh_from_step
from bemkit.geometry.sizing import distance_field, geodesic_field, uniform_field

HERE = Path(__file__).parent
STEP = HERE / "geometry" / "plate.step"
MESH = HERE / "mesh"

# mm -- same ramp as the pre-restructure config.toml this replaces.
LC_MIN, LC_MAX = 3.0, 15.0
DIST_MIN, DIST_MAX = 10.0, 100.0

# Uniform resolutions, picked against lambda = 34 mm at 10 kHz (roughly
# 11 / 7 / 3 elements per wavelength) -- a convergence check independent of
# the sizing ramp.
UNIFORM_MM = (3.0, 5.0, 10.0)


def main():
    MESH.mkdir(exist_ok=True)

    mesh_from_step(STEP, MESH / "euclidean.msh",
                    distance_field(LC_MIN, LC_MAX, DIST_MIN, DIST_MAX))
    mesh_from_step(STEP, MESH / "geodesic.msh",
                    geodesic_field(LC_MIN, LC_MAX, DIST_MIN, DIST_MAX))
    for lc in UNIFORM_MM:
        mesh_from_step(STEP, MESH / f"uniform{int(lc)}.msh", uniform_field(lc))


if __name__ == "__main__":
    main()
