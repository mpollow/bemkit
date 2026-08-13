# plate — geodesic sizing, the simplest demonstration

A finite 200x200 mm plate, 5 mm thick, with a 40 mm vibrating membrane in the
center (`bemgeo.baffled_piston`; committed STEP). Two variants differ only in
the size-ramp metric (`model.py`, `mesh_plate.py`):

    mesh_from_step(STEP, "euclidean.msh", distance_field(3.0, 15.0, 10.0, 100.0))
    mesh_from_step(STEP, "geodesic.msh",  geodesic_field(3.0, 15.0, 10.0, 100.0))

```bash
python models/plate/mesh_plate.py     # generates mesh/*.msh (once; already committed)
bem models/plate check
bem models/plate solve
python models/plate/compare_meshes.py
```

## Why a plate

The back-face center is **5 mm** from the membrane straight through the plate,
but **~100 mm** over the surface (out to an edge, around, and back). So:

- **Euclidean** distance refines a shadow disc on the back face directly behind
  the membrane — 5 mm is well inside `dist_min`.
- **Geodesic** distance leaves the entire back face at `lc_max`, because every
  back-face point is a full plate-width away over the surface.

`compare_meshes.py` prints front-face vs back-face mean element size for both;
the geodesic back face sits near `lc_max` while the euclidean back face is
refined (measured: euclidean back/front ≈ 1.0x, geodesic ≈ 2.4x). This is the
same effect the geodesic algorithm's unit tests check on a folded strip
(`packages/bemkit/tests/geometry/test_geodesic.py`), shown end-to-end through
gmsh.

## Uniform variants — a resolution study

Three extra variants mesh the plate at a single element size everywhere
(`uniform_field(lc)` — no distance ramp at all), for a convergence check
independent of the sizing metric:

    uniform3.msh    lc = 3 mm   (~11 elements per wavelength at 10 kHz)
    uniform5.msh    lc = 5 mm   (~7 elements per wavelength)
    uniform10.msh   lc = 10 mm  (~3 elements per wavelength)

Sizes are picked against λ = 34 mm at 10 kHz. The simulation runs to 16 kHz
(`sr = 32000`, `nbins = 65` in `params.toml`).
