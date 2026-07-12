# How to run a BEM simulation: geometry → results

A complete, worked walkthrough using the `bem` CLI (`bemkit`), from any input
geometry to plotted/queryable acoustic results. The running example is a
vibrating spherical cap (`sphcap`), but the workflow is identical for scanned
heads, CAD assemblies, or any triangulated surface.

The whole pipeline is five verbs:

```
mesh  →  export  →  run  →  pack  →  plot / results / export-web
```

`bem MODEL all` chains `mesh → export → run → plot` in one shot.

---

## 0. Prerequisites (one-time)

```bash
# Install bemkit + all optional deps (gmsh, matplotlib, pyvista, scipy)
pip install -e .[all]
#   ...or a subset: .[meshing] for gmsh, .[viz] for 2D plots, .[viz3d] for 3D,
#   .[analytic] for scipy (sphcap analytic solution)
#   Without installing: use `python -m bemkit.cli ...` instead of `bem ...`

# NumCalc must be on PATH (the actual solver — not part of bemkit)
NumCalc -h        # if "command not found", add its bin/ dir to PATH
```

`bem <MODEL_DIR>` with no subcommand always prints a status report — run it any
time to see what stage each variant is at. `MODEL_DIR` defaults to `.`, so
running `bem <subcommand>` from inside a model directory works too
(`bem mesh` ≡ `bem . mesh`); a model directory that happens to share a name
with a subcommand (e.g. `mesh/`) needs the explicit `./` prefix to disambiguate.

---

## 1. Create the model directory

Everything for one study lives in `examples/<name>/`. The only required file
is `config.toml`. Subdirectories are created as you go:

```
examples/sphcap/
├── config.toml         # ← the only thing you must write by hand
├── geometry/           # remesh sources: CAD (*.step) + scans (*.ply, *.stl, *.obj)
├── mesh/                # trusted meshes, used as-is (*.msh, *.stl, *.ply, *.mesh)
├── fieldpoints/          # where you want to evaluate pressure (*.npy, METERS)
├── pre_*.py|.sh          # optional local preprocessing (run automatically by `mesh`)
├── numcalc/<variant>/    # generated: NumCalc job bundles + results
│                         #   (snapshot.json records the config that produced them)
└── plots/                # generated: figures
```

### `config.toml` — minimal but complete

```toml
sr = 32000            # sample rate (Hz) → sets the frequency grid
nbins = 65            # n_frequencies = nbins - 1  (DC bin excluded)
                      # freqs = linspace(0, sr/2, nbins)[1:]

air_density   = 1.2   # rho0
speed_of_sound = 340.0
bem_type = 4          # 4 = DMLFMBEM, O(N log N)  [recommended default]
time_factor = -1.0    # e^{-iωt} convention
is_internal = 0       # 0 = exterior (radiation) problem

# Mesh sizing (mm). Aim for ~6 elements per wavelength at the highest frequency.
[mesh]
lc_min = 5.0          # finest element edge
lc_max = 25.0         # coarsest element edge (far from the source)
dist_min = 50.0       # size ramps from lc_min→lc_max between these distances
dist_max = 100.0

# Where to evaluate the field. Point to a directory (loads every *.npy as a set)
fieldpoints = "fieldpoints"
# ...or name sets explicitly:
# [fieldpoints]
# onaxis  = "fieldpoints/onaxis.npy"
# polar   = "fieldpoints/polar.npy"
```

> **Units matter.** Fieldpoint `.npy` files are in **meters**. `.msh` meshes are
> **millimeters** (GMSH convention); `.stl`/`.ply`/`.mesh` are **meters**. The
> pipeline converts automatically based on file extension.

---

## 2. Provide geometry — pick ONE of three entry points

`bem MODEL mesh` first runs any `pre_*.py` / `pre_*.sh` scripts in the model dir,
then meshes every source it finds in `geometry/` into a surface `.msh` in `mesh/`.
The split is strict: `geometry/` holds things that still need meshing; anything
in `mesh/` is trusted and used as-is, never remeshed. A source is skipped when a
mesh derived from it already exists (including recipe-suffixed ones like
`name.lc5-25.msh` from a `pre_` script).

### (A) CAD solid — `geometry/*.step`
Drop a STEP file in `geometry/`. GMSH generates a size-graded triangle mesh.
Physical groups come from **STEP face colors**:
- **Blue (0,0,255)** → `RIGID` (zero velocity, sound-hard) — the default
- **Red (255,0,0)** → `VIBRATING` (the driven surface)

```bash
cp my_driver.step examples/sphcap/geometry/
bem examples/sphcap mesh          # → mesh/my_driver.msh
```

### (B) Scanned / tessellated surface to be re-meshed — `geometry/*.{ply,stl,obj}`
A tessellated surface in `geometry/` is re-classified and re-meshed by GMSH
with the same size fields. Scan input is treated as meters and scaled to mm.

```bash
cp head_scan.ply examples/head/geometry/
bem examples/head mesh            # → mesh/head_scan.msh
```

### (C) A mesh you already trust — `mesh/*.{msh,stl,ply,mesh}`
If a usable surface mesh already exists, drop it in `mesh/` — anything there is
used as-is and never remeshed (put a scan in `geometry/` instead if you *do*
want remeshing). The mesh file stem becomes the *variant name*.

```bash
cp left.stl right.stl examples/hutubs/mesh/     # meters
bem examples/hutubs export                       # uses them as-is
```

> **Variants = mesh files.** Every mesh file under `mesh/` becomes a variant with
> its own `numcalc/<stem>/` bundle. One file → one variant; many files → many
> variants meshed/exported/run/packed independently. (You can also define
> BC-only variants that share a mesh via a `[variants]` table — see CLAUDE.md.)

### Multi-variant meshing via a `pre_` script (the `sphcap` pattern)
To sweep several CAD files × several mesh resolutions, script it. `mesh` runs it
for you (files prefixed `__` are skipped):

```python
# examples/sphcap/pre_2_make_mesh_from_step.py
from pathlib import Path
from bemkit.meshing.generate import mesh_from_step

mesh_dir = Path.cwd() / "mesh"; mesh_dir.mkdir(exist_ok=True)
for step in sorted((Path.cwd() / "geometry").glob("*.step")):
    for lc_min, lc_max in [(5, 25), (10, 10)]:
        out = mesh_dir / f"{step.stem}.lc{lc_min}-{lc_max}.msh"
        mesh_from_step(step, out, lc_min=lc_min, lc_max=lc_max,
                       dist_min=50, dist_max=100)
```

---

## 3. Define where to measure — fieldpoints

Fieldpoints are the coordinates (in **meters**) where you want the radiated
pressure. Generate `.npy` files, typically from a `pre_*.py` script so they
regenerate with the mesh:

```python
# examples/sphcap/pre_fieldpoints.py
from pathlib import Path
import numpy as np
from bemkit import make_fields_onaxis, make_fields_polar, make_fields_plane

fp = Path(__file__).parent / "fieldpoints"; fp.mkdir(exist_ok=True)
np.save(fp / "onaxis.npy",  make_fields_onaxis(dim='z', radii=[0.1, 0.25, 0.5, 1.0, 2.0]))
np.save(fp / "polar.npy",   make_fields_polar(dims='zx', res=5))            # circle
np.save(fp / "planeXZ.npy", make_fields_plane(dims='xz',
                                              limits=[-500, 500, -500, 500], res=10))
```

Points inside the mesh (for exterior problems) or too close to the surface are
dropped automatically during export (ray-cast validation).

---

## 4. Drive the surface — boundary conditions

Which elements vibrate is set by STEP color (entry A) or, for entries B/C, by a
`[BOUNDARY]` table in `config.toml`. Element indices are **0-based**; the `TO`
range is inclusive.

The friendliest form is a **geometric anchor**: give a physical point (meters,
like fieldpoints) and the driven elements are found on the mesh at export time.
Anchors survive remeshing — one definition serves every resolution/variant of
the same geometry:

```toml
[BOUNDARY]
earopening = { near = [0.06, 0.0, 0.10] }                # nearest element
cap = { near = [0.0, 0.0, 0.05], radius = 0.01 }         # all elements within 10 mm
spots = { near = [[0.06, 0, 0.1], [-0.06, 0, 0.1]], velocity = 0.5 }  # one per anchor
```

Export fails loudly if an anchor lands implausibly far from the surface or a
`radius` selects nothing (usually a units/frame mistake).

Raw element indices also work — 0-based, inclusive `TO` ranges — but they go
stale on every remesh:

```toml
[BOUNDARY]
earopening = [100, 100, 1.0]        # element 100 driven at v = 1.0 m/s

# Per-variant (different element per mesh):
[BOUNDARY.left]
earopening = [17379, 17379, 1.0]

# A non-contiguous driven patch:
[BOUNDARY.sphcap]
cap = [[0, 24, 1.0], [80, 95, 1.0]]
```

Elements with no BC are implicitly sound-hard (`VELO = 0`). NumCalc also supports
`PRES` (sound-soft) and `ADMI` (absorbing/impedance) — see CLAUDE.md.

### Incident sources (scattering) — plane waves / point sources
Instead of (or in addition to) a vibrating surface, incident fields pass
straight through to `NC.inp` as 8-value rows:

```toml
["PLANE WAVES"]
#     nr, x0, y0, z0, ampl_real, -1, ampl_imag, -1
1 = [  1,  0, -1,  0,     1,     -1,     0,     -1]

["POINT SOURCES"]
1 = [  1,  0, -1,  0,     1,     -1,     0,     -1]
```

Amplitudes are frequency-independent (`CURVES` are not supported; the curve-ref
slots are always written as `-1`). Note the default result normalization
(`pressure_acc`) assumes a velocity BC — for pure scattering setups read
results with `norm="raw"`.

Sanity-check the setup in 3D **before** running:

```bash
bem examples/sphcap geometry          # mesh + driven (red) elements + eval grid
```

---

## 5. Export the NumCalc job bundle(s)

```bash
bem examples/sphcap export
# For each variant, writes a self-contained numcalc/<variant>/ containing:
#   ObjectMesh/  EvalGrid/  NC.inp  eval_offsets.json  snapshot.json
```

`export` = fieldpoints + mesh + `NC.inp`, for every variant. Restrict with
`--variant <prefix>` (repeatable, prefix match), e.g.
`bem examples/sphcap export --variant sphcap-50-10`.

`NC.inp`'s header records the variant name, sample rate/frequency count, mesh/eval-grid
sizes, and fieldpoint set counts, so a bundle is readable without cross-referencing
`config.toml`. `snapshot.json` freezes the effective config used to produce the bundle;
result loaders prefer it over the live `config.toml`, and warn if you've since edited
`sr`/`nbins`/etc. so old results wouldn't line up with the new frequency grid.

---

## 6. Run the solver

Each `numcalc/<variant>/` is a relocatable bundle: NumCalc runs with it as CWD
and writes results into `be.out/`. Frequency bins are independent, so runs
parallelize and resume trivially.

```bash
bem examples/sphcap run                       # all variants, parallel
bem examples/sphcap run -j 4                  # cap at 4 concurrent NumCalc procs
bem examples/sphcap run --split 4             # fan each variant into 4 bin-range jobs
bem examples/sphcap run --split 4 --missing-only   # resume: only redo absent bins
```

`--missing-only` computes the gap from what's already in `be.out/`/`be.npz`, so a
killed run continues where it stopped — no bookkeeping file needed.

> **Convergence ≠ completion.** A bin can finish but not converge. `bem run` and
> `bem status` scan `NC*.out` and surface warnings
> ("Maximum number of iterations is reached", "CGS failed"). These usually mean a
> defective mesh — **re-mesh** rather than raising iteration limits.

Check progress any time:

```bash
bem examples/sphcap                   # status: per-variant bins done / missing / warnings
```

You can also run NumCalc entirely outside bemkit:
`cd numcalc/<variant> && NumCalc` (outputs `NC.out`, `be.out/`).

---

## 7. Pack results (optional but recommended)

```bash
bem examples/sphcap pack                    # be.out/ (ASCII) → be.npz (~4× smaller, complex64)
bem examples/sphcap pack --delete-original  # verify, then remove be.out/
```

`be.npz` shadows `be.out/` everywhere downstream. (A subsequent full `run`
removes a stale `be.npz` so it can't hide fresh output — re-`pack` afterwards.)

---

## 8. Look at the results

### 2D plots → `plots/*.png`
```bash
bem examples/sphcap plot                    # frequency magnitude + impulse responses (all sets)
bem examples/sphcap plot freq
bem examples/sphcap plot polar
bem examples/sphcap plot plane  --freq 1000
bem examples/sphcap plot sphere --freq 1000
bem examples/sphcap plot phase | etc | ir | contour
```

### Interactive 3D field/surface inspection
```bash
bem examples/sphcap results onaxis --freq 1000                 # surface + field pressure
bem examples/sphcap results onaxis --freq 1000 --variant sphcap-50-10
#   --quantity magnitude_db|magnitude|phase|real|imag
#   --intensity  (acoustic intensity arrows)   --impedance   --plane   --screenshot out.png
```

### Export a web auralizer bundle
```bash
bem examples/sphcap export-web --out bundle/            # solved variants → auralizer bundle
bem examples/sphcap export-web --out bundle/ --directions g30_1m --planes planeXZ
```
Consumed by the separate `bemkit-auralizer` framework (format contract lives in that
repo's `docs/bundle-format.md`).

### Programmatic access — `SimulationData`
Numpy-like, lazily-loaded access to every array in a variant directory:

```python
from bemkit import SimulationData

sim = SimulationData.from_model("examples/sphcap")                    # auto-pick variant
sim = SimulationData.from_model("examples/sphcap", variant="sphcap-50-10")

sim.set_names            # ['onaxis', 'polar', 'planeXZ', ...]
sim.frequencies          # (nf,) Hz
sim.available_bins       # [1, 2, ..., 64]

sim.p_onaxis             # (N, nf) complex — normalized pressure (÷ rho0·ω, the default)
sim.p_raw_onaxis         # (N, nf) complex — raw Pascals
sim.nodes_onaxis         # (N, 3) meters
sim.boundary_pressure()  # (M, nf) surface pressure
result = sim.result("onaxis")   # BEMResult bridge for the viz functions
```

---

## The whole thing, end to end

```bash
# 1. author examples/sphcap/config.toml, drop geometry in geometry/ or mesh/,
#    add pre_fieldpoints.py (+ BC in config.toml or STEP colors)

bem examples/sphcap geometry          # 2. eyeball mesh + BCs + eval grid (3D)
bem examples/sphcap all               # 3. mesh → export → run → plot
#    ...or step through:
#    bem examples/sphcap mesh
#    bem examples/sphcap export
#    bem examples/sphcap run -j 4
#    bem examples/sphcap pack
#    bem examples/sphcap plot

bem examples/sphcap                   # 4. status: everything complete, no warnings?
bem examples/sphcap results onaxis --freq 1000   # 5. inspect the field in 3D
```
