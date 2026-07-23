# plate — geodesic sizing, the simplest demonstration

A finite 200x200 mm plate, 5 mm thick, with a 40 mm vibrating membrane in the
center (`bemgeo.baffled_piston`; committed STEP). Two variants differ only in
the size-ramp metric:

    [variants.euclidean]  mesh = { distance = "euclidean" }
    [variants.geodesic]   mesh = { distance = "geodesic" }

```bash
bem examples/plate mesh
python examples/plate/compare_meshes.py
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
refined. This is the same effect the geodesic algorithm's unit tests check on a
folded strip (`tests/meshing/test_geodesic.py`), shown end-to-end through gmsh.

## Uniform variants — a resolution study

Three extra variants mesh the plate at a single element size everywhere
(`lc_min = lc_max`), for a convergence check independent of the sizing ramp:

    [variants.uniform3]   mesh = { lc_min = 3.0,  lc_max = 3.0 }
    [variants.uniform5]   mesh = { lc_min = 5.0,  lc_max = 5.0 }
    [variants.uniform10]  mesh = { lc_min = 10.0, lc_max = 10.0 }

Sizes are picked against λ = 34 mm at 10 kHz (≈ 11 / 7 / 3 elements per
wavelength). The simulation runs to 16 kHz (`sr = 32000`, `nbins = 65`).
