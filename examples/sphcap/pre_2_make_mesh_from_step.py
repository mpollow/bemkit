#!/usr/bin/env python3
"""Generate mesh variants from all STEP files × mesh resolutions."""

from pathlib import Path
from pynumcalc.meshing.generate import mesh_from_step

model_dir = Path.cwd()
mesh_dir = model_dir / "mesh"
mesh_dir.mkdir(exist_ok=True)

params = [
    (5, 25, 50, 100),
    (5,  5, 50, 100),
    (10, 10, 50, 100),
    (10, 25, 50, 100),
]

for step_file in sorted((model_dir / "geometry").glob("*.step")):
    for lc_min, lc_max, dist_min, dist_max in params:
        name = f"{step_file.stem}.lc{lc_min}-{lc_max}"
        output = mesh_dir / f"{name}.msh"
        print(f"Meshing {step_file.name} → {output.name}  (lc={lc_min}/{lc_max} dist={dist_min}/{dist_max})")
        mesh_from_step(step_file, output,
                       lc_min=lc_min, lc_max=lc_max,
                       dist_min=dist_min, dist_max=dist_max)
