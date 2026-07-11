# bemkit

Public, GPL-3.0-licensed repo. Python library and `bem` CLI for Mesh2HRTF NumCalc
simulations from CAD geometry.

## Project Structure

```
bemkit/
├── bemkit/
│   ├── __init__.py     # eager core exports; analysis/viz resolve lazily via __getattr__
│   ├── config.py       # ModelConfig (single config object, dict-style access), freq_bin
│   ├── pipeline.py      # PipelineStage ABC + Mesh/Export/Run/Pack stages + STAGES dict
│   ├── meshing/          # io.py    — mesh file I/O, element_areas, physical groups
│   │                      # generate.py — gmsh STEP/PLY meshing (lazy gmsh)
│   ├── sim/              # NumCalc layer (no analysis/viz deps)
│   │                      # fields.py      — make_fields_onaxis/polar/plane generators
│   │                      # fieldpoints.py — validation, loading, offsets
│   │                      # writers.py     — NumCalc input writing (write_nc_inp,
│   │                      #                  templates/NC.inp package data)
│   │                      # export.py      — export_variant, ExportInfo
│   │                      # readers.py     — raw NumCalc output parsers, load_bins
│   │                      # runner.py      — Runner ABC, LocalRunner, plan_jobs, run_jobs,
│   │                      #                  split_steps, missing_ranges
│   │                      # status.py      — available_bins/missing_bins/is_complete,
│   │                      #                  solver_warnings (NC*.out health scan)
│   │                      # packing.py     — be.out/ <-> be.npz, clear_pack
│   │                      # discovery.py   — discover_variants, numcalc_dir
│   ├── analysis/         # Results layer (depends on sim, not vice versa)
│   │                      # simulation.py  — SimulationData lazy result accessor
│   │                      # result.py      — BEMResult, load_result
│   │                      # normalization.py — normalize_pressure
│   │                      # _grid.py       — grid detection (private)
│   │                      # analytic.py    — spherical cap solver (optional: scipy)
│   ├── viz/               # 2D/3D visualization (matplotlib/pyvista)
│   │                      # freq.py, time.py, polar.py, field.py — 2D plots
│   │                      # scene.py       — BEMScene 3D rendering (lazy pyvista, no I/O)
│   │                      # results_viewer.py — interactive 3D viewer (owns data loading)
│   │                      # _style.py      — colormaps, formatting
│   ├── web.py             # auralizer-bundle exporter (`bem MODEL export-web`);
│   │                      #   format contract: bemkit-auralizer docs/bundle-format.md
│   └── cli/               # `bem` entry point
│                          # _pipeline.py — mesh/export/run/pack/all + status
│                          # _plot.py     — 2D plot subcommand
│                          # _viz3d.py    — geometry/results 3D subcommands
│                          # _web.py      — export-web subcommand
├── pyproject.toml      # deps: numpy + meshio only; extras: viz/viz3d/meshing/analytic/all
└── examples/              # Per-model dirs (see below)
```

**Layer rule**: `meshing`/`sim` (core) never import `analysis`/`viz`, and never use
pyvista/matplotlib at all — fieldpoint validation is pure numpy (Möller-Trumbore +
Eberly kernels). matplotlib/pyvista are only imported lazily inside `analysis`/`viz`/`cli`
commands that need them (`plot`, `geometry`, `results`). Enforced by `tests/test_boundaries.py`.

**Scope**: public meshing is gmsh STEP/PLY generation; public examples are models whose
surface is rigid or vibrating (the implemented STEP color contract). Fieldpoint generators
are the minimal set: `onaxis`, `polar`, `plane`. The mmgs metric workflow and gaussian
spherical grids live in the private layer.

**Geometry authoring** lives in the external `bemgeo` package (separate repo),
not in bemkit. bemkit is a pure consumer of the STEP files bemgeo writes: the color
contract (blue=rigid, red=vibrating) is *read* in `bemkit/meshing/generate.py`.

`pip install -e .` installs the `bem` console script (entry point `bemkit.cli:main`).
Without installing (e.g. externally-managed system Python), use `python -m bemkit.cli`
— identical interface: `python -m bemkit.cli examples/sphcap export`.

Model directory layout (unchanged):
```
examples/<name>/
├── config.toml          # Configuration
├── pre_*.py / pre_*.sh  # Model-local preprocessing scripts
├── fieldpoints/       # .npy fieldpoint files
├── geometry/          # CAD output (.step)
├── mesh/              # Mesh files (.msh, .stl, .ply)
├── numcalc/<variant>/ # Per-variant: ObjectMesh/, EvalGrid/, NC.inp, NC.out, be.out/ (or be.npz), eval_offsets.json
└── plots/             # Output figures
```

## Usage

### Pipeline CLI (`bem`)

MODEL_DIR defaults to `.` (the current directory) when the first token is a
known subcommand or absent, so `bem mesh` from inside a model directory is
equivalent to `bem . mesh`. A model directory literally named the same as a
subcommand (e.g. `mesh/`) needs a path prefix to disambiguate: `bem ./mesh`.

```bash
bem examples/sphcap                    # show status (default)
bem examples/sphcap mesh               # generate mesh from geometry/*.step or mesh/*.ply
bem examples/sphcap export             # fieldpoints + mesh + NC.inp (all variants)
bem examples/sphcap run                # run NumCalc (parallel)
bem examples/sphcap run -j 4           # cap parallel NumCalc instances
bem examples/sphcap run --split 4 --missing-only  # fan out + resume (see below)
bem examples/sphcap pack               # be.out/ → be.npz (~4× shrink, complex64)
bem examples/sphcap pack --delete-original   # also remove be.out/ after verify
bem examples/sphcap plot               # freq + ir plots (all sets)
bem examples/sphcap plot freq          # frequency magnitude
bem examples/sphcap plot ir
bem examples/sphcap plot polar
bem examples/sphcap plot plane --freq 1000
bem examples/sphcap plot phase
bem examples/sphcap plot etc
bem examples/sphcap plot sphere --freq 1000
bem examples/sphcap plot contour --freq 1000
bem examples/sphcap all                # mesh -> export -> run -> plot
# --variant goes AFTER the subcommand:
bem examples/sphcap export --variant foo
bem examples/sphcap run --variant foo

# Run NumCalc externally: cd numcalc/{variant} && NumCalc  (output: NC.out, be.out/)

# 3D pre-run inspection (was viz_geometry.py)
bem examples/sphcap geometry                              # mesh + BC + eval grid
bem examples/sphcap geometry --variant sphcap-50-20        # prefix match, slider if >1

# 3D post-run inspection (was viz_results.py)
bem examples/sphcap results onaxis --freq 1000            # surface + field pressure
bem examples/sphcap results onaxis --freq 1000 --variant sphcap-50-20  # variant filter
```

`bem MODEL export-web` writes an auralizer web bundle (consumed by the
bemkit-auralizer framework; the format contract is that repo's
docs/bundle-format.md, guarded there by a golden-manifest test). The
diagnostic `bem MODEL check ...` family ships separately via the
`bemkit.cli` entry-point group (private `bemlab` package).

`bem MODEL geometry` options: `--variant` (repeatable), `--no-bc`, `--no-mesh`, `--no-grid`,
`--mesh-opacity`, `--mesh-color`, `--grid-color`, `--point-size`, `--show-edges`,
`--bc-vibrating-color`, `--camera` (xz/xy/yz/iso), `--background`, `--screenshot`

`bem MODEL results [SET]` options: `--variant` (repeatable), `--quantity` (magnitude_db/magnitude/phase/real/imag),
`--surface-quantity`, `--points-quantity`, `--norm` (acc/raw), `--no-surface`, `--no-points`,
`--no-phase-fade`, `--plane`, `--intensity`, `--intensity-scale`, `--intensity-length` (db/magnitude/uniform),
`--intensity-clim LO HI`, `--impedance`, `--impedance-quantity`, `--mesh-opacity`, `--point-size`,
`--camera` (xz/xy/yz/iso), `--background`, `--screenshot`

### SimulationData (lazy result accessor)

`SimulationData` (in `bemkit.analysis.simulation`, exported from package root) provides
numpy-like access to all data in a `numcalc/{variant}/` directory with lazy loading (files
read on first access, then cached).

```python
from bemkit import SimulationData

sim = SimulationData.from_model("examples/sphcap")          # auto-discover variant
sim = SimulationData.from_model("examples/sphcap", variant="sphcap-50-20.lc5-25")  # explicit variant

sim.set_names             # ['onaxis', 'polar', 'planeXZ', ...]
sim.frequencies           # (nf,) Hz array
sim.available_bins        # [1, 2, ..., 64]

# Fieldpoint data — dynamic attrs: sim.p_<set>, sim.v_<set>, sim.nodes_<set>
sim.p_onaxis              # (N, nf) complex — normalized pressure (pressure_acc, default)
sim.p_raw_onaxis          # (N, nf) complex — raw pressure (velocity BC output)
sim.v_onaxis              # (N, nf, 3) complex — normalized 3D velocity (velocity_acc, default)
sim.v_raw_onaxis          # (N, nf, 3) complex — raw 3D velocity (velocity BC output)
sim.nodes_onaxis          # (N, 3) — coordinates in meters
# Or via methods:
sim.pressure("onaxis")              # normalized (norm="acc" default)
sim.pressure("onaxis", norm="raw")  # raw pressure
sim.velocity("onaxis")              # normalized 3D velocity (norm="acc" default)
sim.velocity("onaxis", norm="raw")  # raw 3D velocity
sim.nodes("onaxis")                 # coordinates

# Boundary (surface mesh)
sim.boundary_pressure()             # (M, nf) complex — normalized surface pressure (norm="acc" default)
sim.boundary_pressure(norm="raw")   # (M, nf) complex — raw surface pressure
sim.boundary_velocity()             # (M, nf) complex — normalized normal velocity (norm="acc" default)
sim.boundary_velocity(norm="raw")   # (M, nf) complex — normal velocity (raw)
sim.boundary_pressure_acc           # (M, nf) alias for boundary_pressure(norm="acc")
sim.boundary_velocity_acc           # (M, nf) alias for boundary_velocity(norm="acc")
sim.boundary_nodes                  # (M, 3)
sim.boundary_elements               # (T, 3) int — triangle connectivity

# BEMResult bridge (for existing viz functions)
result = sim.result("onaxis")  # returns BEMResult with pressure_acc normalized

# Single-bin access (no full load, for interactive viewers)
sim.pressure_at(12, domain="boundary")  # (M,) complex
sim.velocity_at(12, domain="eval")      # (N_total, 3) complex
```

## Configuration: config.toml

Per-model config in `examples/<name>/config.toml`. `modelname` is optional (defaults to directory name).

```toml
modelname = "sphcap"      # optional
sr = 48000
nbins = 65                # n_frequencies = nbins - 1 (DC excluded)
air_density = 1.2
speed_of_sound = 340.0
bem_type = 4              # 0=TBEM O(N³), 1=SLFMBEM O(N^1.5), 4=DMLFMBEM O(N log N) [default]
time_factor = -1.0
is_internal = 0

[cad]
radius = 50               # mm
aperture = 20             # degrees

[mesh]
lc_min = 5.0              # mm — ~6 elements per wavelength at max freq
lc_max = 25.0
dist_min = 50.0
dist_max = 100.0

[fieldpoints]
onaxis = "fieldpoints/onaxis.npy"   # file path, relative to model dir (or absolute)
polar = "fieldpoints/polar.npy"
plane = "fieldpoints/plane.npy"
```

Fieldpoints config can also be a plain directory path: `fieldpoints = './fieldpoints'`
(loads all `.npy` files in that directory, one set per file).

Each entry must be a `.npy` file path (or the whole `[fieldpoints]` table replaced by a
directory string) — there is no expression syntax.

### Generating fieldpoint .npy files (bemkit.sim.fields)

`make_fields_onaxis`, `make_fields_polar`, `make_fields_plane` are Python helpers (exported
from the package root) for *generating* `.npy` files — they are not referenced from
`config.toml` directly. Use them in a `pre_*.py` script and `np.save()` the result into
`fieldpoints/`:

```python
from bemkit import make_fields_polar
import numpy as np

points = make_fields_polar(dims='xy', res=5, radius=1.0)  # (N, 3) meters
np.save("examples/sphcap/fieldpoints/polar.npy", points)
```

- `make_fields_onaxis(dim='x', radii=[...])` — points along one axis (meters)
- `make_fields_polar(dims='xy', res=5, radius=1.0)` — circle in a plane (meters)
- `make_fields_plane(dims='xy', res=5, limits=[-200, 200, -200, 200])` — grid on a plane
  (res/limits in mm)

## Mesh Variants

All NumCalc data lives under `numcalc/{variant}/` where variant = mesh file stem. `bem MODEL export`
(and `mesh`/`run`/`pack`) auto-discover mesh files in `mesh/` and process each.

- Single mesh (e.g. `mesh/sphcap_10.msh`) → `numcalc/sphcap_10/`
- Multiple meshes (e.g. `mesh/left.meshA.stl`, `mesh/right.meshB.stl`) → `numcalc/left.meshA/`, `numcalc/right.meshB/`

Fieldpoint offsets: `numcalc/{variant}/eval_offsets.json`. Results read directly from
`be.out/` (or packed `be.npz`) via `BEMResult.from_numcalc()` / `SimulationData`.

### BC-defined variants (same mesh, different boundary conditions)

Use a `[variants]` table to define variant names explicitly. Each entry can have an optional `mesh` key pointing to the mesh file stem (defaults to the variant name). This allows running the same mesh with different `[BOUNDARY]` sections.

```toml
[variants.left]
mesh = "hutubs21"   # mesh file stem; defaults to variant name if omitted

[variants.right]
mesh = "hutubs21"

[BOUNDARY.left]
earopening = [17379, 17379, 1.0]

[BOUNDARY.right]
earopening = [17387, 17387, 1.0]
```

When `[variants]` is present, `bem` uses it instead of scanning `mesh/`.

## Runner / Remote Execution

A variant directory `numcalc/{variant}/` is a self-contained, relocatable job bundle: NumCalc
runs with it as CWD using only relative paths, and writes results back into it. Frequency
bins are independent (`-istart`/`-iend`), and `be.out/be.{i}/` outputs are disjoint, so one
variant can be split across several processes/machines and merged trivially (no merge step
needed — outputs just land in the same `be.out/`).

- `Runner` ABC (`bemkit.sim.runner`): `submit(job)` / `poll(handle)` / `fetch(job)`.
  `LocalRunner` is the only implementation today (subprocess with CWD=variant dir); SSH or
  scheduler-based runners would implement the same interface — transfer the bundle, start
  NumCalc, report status, bring `be.out/` back.
- `Job(name, nc_dir, steps=(istart, iend) | None)` — one NumCalc invocation, optionally
  scoped to a 1-based inclusive bin range.
- `bem MODEL run --split N` fans each variant out into N bin-range jobs (`-istart`/`-iend`),
  via `plan_jobs`/`split_steps`.
- `bem MODEL run --missing-only` only (re-)runs frequency bins absent from `be.out/`/`be.npz`
  — uses `bemkit.sim.status.missing_bins(nc_dir, nf)` to compute the gap, so partial/failed
  runs can resume without redoing completed bins.
- `bemkit.sim.status`: `available_bins(nc_dir)`, `missing_bins(nc_dir, nf)`,
  `is_complete(nc_dir, nf)` — derive job state directly from `be.out/`/`be.npz`, no separate
  bookkeeping file.
- `be.npz` shadows `be.out/` everywhere, so `bem MODEL run` removes a leftover `be.npz`
  for every variant it runs (with `--missing-only` its bins are first restored into
  `be.out/` so the resume merges); re-run `pack` afterwards. A full recompute (no
  `--missing-only`) also deletes the variant's old `NC*.out` logs so superseded
  warnings don't linger in `bem status`; resumes keep them.
- Bin presence ≠ convergence: NumCalc records non-convergence only in its log.
  `bemkit.sim.status.solver_warnings(nc_dir)` scans `NC*.out` for the two health strings
  ("Maximum number of iterations is reached", "Iteration method CGS failed") and maps them
  to bins; `bem status` and `bem run` surface them. Such warnings usually mean a defective
  mesh — re-mesh rather than raising `-nitermax`.

## Fieldpoint Validation

Points validated against mesh surface using Moller-Trumbore ray-casting (multi-direction majority voting):
- External problems (`is_internal=0`): points inside mesh dropped
- Internal problems (`is_internal=1`): points outside mesh dropped
- Points too close to surface dropped (threshold = `min_surface_distance_factor` x avg_edge_length, default factor 0.1)

## Unit Conventions

| File Type | Units |
|-----------|-------|
| Fieldpoint `.npy` | **METERS** |
| Mesh `.msh` (GMSH) | **MILLIMETERS** |
| Mesh `.stl`/`.ply` | **METERS** |
| NumCalc exports | **METERS** (converted via `get_mesh_scale()`) |
| BEMResult | **METERS** |

`get_mesh_scale()` returns 0.001 for `.msh`, 1.0 for others. `find_mesh_file(mesh_dir, name)` tries `.msh`/`.stl`/`.ply`/`.mesh` in order.

## NC.inp Format

See `bemkit/sim/templates/NC.inp` (package data). Physical group IDs: `RIGID=1` (zero velocity, default), `VIBRATING=2` (specified velocity).
STEP colors: Blue (0,0,255) = rigid, Red (255,0,0) = vibrating.

### NumCalc Boundary Condition Types

Each element can be assigned one of three BC types via `ELEM e1 TO e2 bctype real(v0) curve1 imag(v0) curve2`:

| Type | Effect | v0=0 meaning |
|------|--------|--------------|
| `VELO` | Fixed particle velocity | Sound-hard (rigid) — default if no BC assigned |
| `PRES` | Fixed sound pressure | Sound-soft |
| `ADMI` | Acoustic admittance (1/impedance) | Sound-absorbing material |

- Elements with no assigned BC are implicitly sound-hard (VELO=0).
- An element may have both `VELO` and `ADMI` simultaneously (unique combination).
- `ADMI` supports frequency-dependent values via piecewise-linear curves: `curve1`/`curve2` reference curve numbers for real/imaginary parts of the admittance. Set to `-1` if not frequency-dependent.

### BOUNDARY section in config.toml

Global (same element for all variants):
```toml
[BOUNDARY]
earopening = [start_elem, end_elem, velocity]  # e.g. [100, 100, 1.0]
```

Per-variant (different element per mesh file):
```toml
[BOUNDARY.variant_name]
earopening = [start_elem, end_elem, velocity]

[BOUNDARY.other_variant]
earopening = [start_elem, end_elem, velocity]
```

Multi-range BC (e.g. a cap spanning non-contiguous element indices):
```toml
[BOUNDARY.variant_name]
cap = [[s1, e1, 1.0], [s2, e2, 1.0], ...]   # list of [start, end, velocity] ranges
```

Element indices are 0-indexed. Single element: `[7875, 7875, 1.0]` → `ELEM 7875 TO 7875 VELO 1.0000 -1 0.0 -1`.

## Analytic Solution

`bemkit.analysis.analytic` (exported as `sphcap_pressure` from the package root) computes
exact pressure for axisymmetric spherical cap vibration via SH expansion.
`sphcap_pressure(eval_points, frequencies, c0, rho0, angle_deg, radius_mm, nmax)` → raw pressure array.
Normalization matches BEM pipeline: velocity BC (v=1) → raw pressure → ÷(rho0·omega·A_vib) → pressure_acc.

## Visualization: bemkit.viz

Composable matplotlib/PyVista visualizations. All 2D functions take an Axes and BEMResult, return artists.

2D plots: `freq_magnitude`, `freq_phase`, `freq_real_imag`, `impulse_response`, `energy_time_curve`, `polar_directivity`, `polar_balloon_slice`, `heatmap`, `contour`, `scatter_field`, `spherical_heatmap`

3D scenes (`BEMScene`, in `bemkit.viz.scene`): Pure rendering — all methods accept pre-loaded numpy arrays, no file I/O.
- `add_mesh(nodes, elements)` — generic triangular mesh
- `add_object_mesh(nodes, elements, bc_elements=None)` — ObjectMesh with optional BC coloring
- `add_eval_grid(points)` — evaluation point cloud
- `add_field_points(result, freq)` — BEMResult eval points colored by value
- `add_surface_pressure(nodes, elements, scalars, rgba=False)` — surface colored by pre-computed scalars
- `add_intensity_arrows(points, pressure, velocity)` — acoustic intensity vectors (computes I = ½Re(p·v*))
- `add_impedance_points(points, scalars)` — eval points colored by pre-computed impedance
- Quantities: `magnitude_db`, `magnitude`, `phase`, `real`, `imag`
- Convenience methods: `add_screenshot_hotkey()`, `add_variant_slider()`, `show_or_screenshot()`

`ResultsViewer` (`bemkit.viz.results_viewer`): Interactive viewer that owns data loading, frequency/variant sliders, and clim key bindings. Loads NumCalc files on demand and passes data to BEMScene. Backs `bem MODEL results`.

Point selection: `points=None` (all), int, slice, array, or bool mask.

## Reference

NumCalc source: `mesh2hrtf/NumCalc/src/NC_Input.cpp` in
https://github.com/Any2HRTF/Mesh2HRTF
