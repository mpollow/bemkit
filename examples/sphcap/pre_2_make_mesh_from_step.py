#!/usr/bin/env python3
"""Generate the example mesh from the STEP geometry (same as `bem . mesh`)."""

from pathlib import Path
from bemkit.meshing.generate import mesh_from_step

model_dir = Path.cwd()
mesh_dir = model_dir / "mesh"
mesh_dir.mkdir(exist_ok=True)

lc_min, lc_max, dist_min, dist_max = 5, 25, 50, 100

for step_file in sorted((model_dir / "geometry").glob("*.step")):
    name = f"{step_file.stem}.lc{lc_min}-{lc_max}"
    output = mesh_dir / f"{name}.msh"
    print(f"Meshing {step_file.name} → {output.name}  (lc={lc_min}/{lc_max} dist={dist_min}/{dist_max})")
    mesh_from_step(step_file, output,
                   lc_min=lc_min, lc_max=lc_max,
                   dist_min=dist_min, dist_max=dist_max)
