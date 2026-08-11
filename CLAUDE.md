# bemkit

Public, GPL-3.0-licensed repo. Python library and `bem` CLI for Mesh2HRTF NumCalc
simulations from CAD geometry.

## Project Structure

```
bemkit/
├── bemkit/
│   ├── __init__.py     # curated user API only (plumbing stays importable from
│   │                      #   submodules); analysis/viz resolve lazily via __getattr__
│   ├── config.py       # ModelConfig (typed config object; from_dir/from_dict,
│   │                      #   validates every known section incl. BOUNDARY/SYMMETRY/
│   │                      #   sources), freq_bin, freq_grid, resolve_bins
│   ├── pipeline.py      # PipelineStage ABC + Mesh/Export/Run/Pack stages + STAGES dict
│   ├── meshing/          # io.py    — mesh file I/O, element_areas, physical groups
│   │                      # generate.py — gmsh STEP/scan meshing (lazy gmsh):
│   │                      #   mesh_from_step, mesh_from_surface (.ply/.stl/.obj)
│   │                      # geodesic.py — over-surface distance for sizing
│   │                      #   (pure numpy/scipy, no gmsh; STEP path only)
│   │                      # sizing.py — size-field builders (uniform/threshold/
│   │                      #   geodesic two-pass), sizing_from_config
│   ├── numcalc/          # NumCalc format + process layer. Imports NOTHING from
│   │                      #   bemkit; stdlib + numpy only (see Layer rule).
│   │                      # deck.py        — Deck: the whole NC.inp as one object,
│   │                      #   read()/write(); BC/Source/Curve records; parse_bc_line
│   │                      # meshfiles.py   — ObjectMesh//EvalGrid Nodes/Elements
│   │                      #   writers (take arrays, never mesh paths)
│   │                      # readers.py     — raw NumCalc output parsers, load_bins
│   │                      # runner.py      — Runner ABC, LocalRunner, plan_jobs (auto
│   │                      #                  fan-out/ordering), run_jobs, missing_ranges
│   │                      # status.py      — available_bins/missing_bins/is_complete,
│   │                      #                  solver_warnings (NC*.out health scan)
│   │                      # packing.py     — be.out/ <-> be.npz, clear_pack
│   ├── sim/              # Simulation setup: bemkit's opinions (may import numcalc)
│   │                      # builder.py     — deck_from_config: ModelConfig → Deck
│   │                      #   (FFT freq grid, group→VELO, resolved BC ranges)
│   │                      # fields.py      — make_fields_onaxis/polar/plane generators
│   │                      # fieldpoints.py — validation, loading, offsets
│   │                      # boundary.py    — BC resolution (near= anchors → ranges)
│   │                      # export.py      — export_variant, ExportInfo
│   │                      # discovery.py   — discover_variants, numcalc_dir
│   │                      # config_snapshot.py — snapshot.json per bundle
│   │                      #   (effective config at export; drift warning)
│   ├── analysis/         # Results layer (depends on sim, not vice versa)
│   │                      # simulation.py  — SimulationData lazy result accessor
│   │                      #   (the public results API; raw norm default)
│   │                      # result.py      — BEMResult adapter for viz (internal)
│   │                      # normalization.py — normalize_pressure
│   │                      # _grid.py       — grid detection (private)
│   │                      # analytic.py    — spherical cap solver (optional: scipy)
│   ├── viz/               # 2D/3D visualization (matplotlib/pyvista)
│   │                      # freq.py, time.py, polar.py, field.py — 2D plots
│   │                      # directivity.py — isobaric map (angle x freq)
│   │                      # _angles.py    — circular/on-axis set geometry
│   │                      # _symmetry.py  — ghost geometry for [SYMMETRY]
│   │                      #   models in the 3D viewers (mirror_mesh-derived)
│   │                      # scene.py       — BEMScene 3D rendering (lazy pyvista, no I/O)
│   │                      # results_viewer.py — interactive 3D viewer (owns data loading)
│   │                      # _style.py      — colormaps, formatting
│   ├── web.py             # auralizer-bundle exporter (`bem MODEL export-web`);
│   │                      #   format contract: bemkit-auralizer docs/bundle-format.md
│   └── cli/               # `bem` entry point
│                          # _init.py     — `bem init` config.toml bootstrap
│                          # _pipeline.py — mesh/export/run/pack/all + status
│                          # _plot.py     — 2D plot subcommand
│                          # _viz3d.py    — geometry/results 3D subcommands
│                          # _web.py      — export-web subcommand
├── pyproject.toml      # deps: numpy + meshio only; extras: viz/viz3d/meshing/analytic/all
└── examples/              # Per-model dirs (see below)
```

**Layer rule**: `meshing`/`sim`/`numcalc` (core) never import `analysis`/`viz`, and never
use pyvista/matplotlib at all — fieldpoint validation is pure numpy (Möller-Trumbore +
Eberly kernels). matplotlib/pyvista are only imported lazily inside `analysis`/`viz`/`cli`
commands that need them (`plot`, `geometry`, `results`). Enforced by `tests/test_boundaries.py`.

**Solver knobs and the `[nc]` hatch**: the four `[bem]` solver keys map onto `Deck`
fields (`solver` → `Deck.isolver`; the rest share their name), and `[nc]` sets any
scalar `Deck` field directly, applied last. Derived fields (frequencies, boundary,
sources, symmetry, mesh counts) are rejected there by name, pointing at the section
they really come from. Two combinations NumCalc accepts and then silently ignores —
`solver = 4` with an FMM `bem_type`, and `preconditioner` alongside `solver = 4` —
are rejected at config load and again in `Deck.validate`.

**`numcalc` is stricter**: it imports *nothing* from `bemkit` and depends only on stdlib
+ numpy (no meshio/gmsh/scipy). `sim` may import `numcalc`; never the reverse. The split
is what everything NumCalc-shaped goes through — `deck_from_config` (`sim/builder.py`)
holds every bemkit opinion, `Deck` (`numcalc/deck.py`) holds only what the solver reads.
Adding a NumCalc setting is therefore one `Deck` field plus one line each in
`write()`/`read()`, not six edits across four layers. The rule is also what keeps the
layer extractable as a standalone `pynumcalc` package — see
`docs/numcalc-feature-coverage.md` §4 for the decision and §4.5 for the trigger.

**Scope**: public meshing is gmsh generation from STEP CAD and tessellated scan
surfaces (`geometry/` holds remesh sources; `mesh/` files are trusted as-is);
public examples are models whose
surface is rigid or vibrating (the implemented STEP color contract). Fieldpoint generators
are the minimal set: `onaxis`, `polar`, `plane`. The mmgs metric workflow and gaussian
spherical grids live in the private layer.

**Geometry authoring** lives in the external `bemgeo` package (separate repo),
not in bemkit. bemkit is a pure consumer of the STEP files bemgeo writes: the color
contract (blue=rigid, red=vibrating) is *read* in `bemkit/meshing/generate.py`.

`pip install -e .` installs the `bem` console script (entry point `bemkit.cli:main`).
Without installing (e.g. externally-managed system Python), use `python -m bemkit.cli`
— identical interface: `python -m bemkit.cli examples/sphcap export`.

Model directory layout:
```
examples/<name>/
├── config.toml        # Configuration (the only required file)
├── *.py / *.sh        # Optional scripts, referenced via exec = ... in config.toml
├── fieldpoints/       # Optional pre-made .npy fieldpoint files
├── geometry/          # remesh sources: CAD (.step) + scans (.ply/.stl/.obj)
├── mesh/              # Mesh files (.msh, .stl, .ply)
├── numcalc/<variant>/ # Per-variant: ObjectMesh/, EvalGrid/, NC.inp, NC.out, be.out/ (or be.npz), eval_offsets.json
└── plots/             # Output figures
```

Scripts are never discovered implicitly (no `pre_*` glob): `[geometry].exec`
runs at `bem mesh`, `[fieldpoints].exec` at `bem export`, both with
cwd = model dir and the repo root prepended to PYTHONPATH.

## Usage

### Pipeline CLI (`bem`)

MODEL_DIR defaults to `.` (the current directory) when the first token is a
known subcommand or absent, so `bem mesh` from inside a model directory is
equivalent to `bem . mesh`. A model directory literally named the same as a
subcommand (e.g. `mesh/`) needs a path prefix to disambiguate: `bem ./mesh`.

```bash
bem examples/sphcap                    # show status (default; `bem MODEL status` is the same)
#   status shows per-variant stages, bins done, solver warnings, and live
#   NumCalc jobs (⟳ running, current bin — detected via /proc CWDs)
bem newmodel init model.msh            # bootstrap config.toml from an existing mesh
#   Copies the mesh into mesh/ (or uses what is already there) and writes a
#   config.toml measured against it: sr=32000/nbins=65, onaxis fieldpoints
#   placed clear of the bounding box, and — only when the mesh has no
#   'vibrating' physical group — a placeholder [BOUNDARY] near= anchor, since
#   a mesh with neither exports as a fully rigid surface that solves to zero.
#   Trusted meshes need no [mesh]/[variants] section, so none is written.
#   Writes config.toml only; run `export` then `run` next. --force overwrites.
bem examples/sphcap mesh               # mesh geometry/ sources (.step CAD, .ply/.stl/.obj scans)
bem examples/sphcap export             # fieldpoints + mesh + NC.inp (all variants)
bem examples/sphcap run                # run NumCalc in the background, return immediately
bem examples/sphcap run -j 4           # at most 4 NumCalc processes (default 1; auto bin fan-out)
bem examples/sphcap run --force        # recompute all bins (default only runs missing ones)
bem examples/sphcap run --foreground   # wait for it (what `all` and CI want)
bem examples/sphcap run --restart      # kill the running NumCalc first, then resume
bem examples/sphcap run --preview 8    # 8 bins across the band, log-spaced
bem examples/sphcap run --freq 1000 4000   # only the bins nearest these
bem examples/sphcap run --bins 4 16 64     # literal 1-based bins
#   A partial run merges into a later full run: `bem MODEL run` then computes
#   only what is still missing. --force cannot be combined with these (it
#   discards every bin, which would mix exports).
bem examples/sphcap pack               # be.out/ → be.npz (~4× shrink), then remove be.out/
bem examples/sphcap pack --keep-original     # keep be.out/ after the verified pack
bem ps                                 # every running NumCalc, any model: model/variant/bins/pid
bem examples/sphcap plot               # freq + ir plots (all sets)
bem examples/sphcap plot freq          # frequency magnitude (off-axis fan on a circular set)
bem examples/sphcap plot ir
bem examples/sphcap plot polar
bem examples/sphcap plot isobaric      # angle x frequency map (circular set)
bem examples/sphcap plot onaxis        # distance-compensated response (on-axis set)
bem examples/sphcap plot plane --freq 1000
bem examples/sphcap plot phase
bem examples/sphcap plot etc
bem examples/sphcap plot sphere --freq 1000
bem examples/sphcap plot contour --freq 1000
bem examples/sphcap all                # mesh -> export -> run
# --variant goes AFTER the subcommand:
bem examples/sphcap export --variant foo
bem examples/sphcap run --variant foo

# Run NumCalc externally: cd numcalc/{variant} && NumCalc  (output: NC.out, be.out/)

# 3D pre-run inspection (was viz_geometry.py)
bem examples/sphcap geometry                              # mesh + BC + eval grid
bem examples/sphcap geometry --variant fine                # prefix match, slider if >1

# 3D post-run inspection (was viz_results.py)
bem examples/sphcap results --freq 1000                   # surface + all fieldpoint sets
bem examples/sphcap results onaxis --freq 1000            # restrict to one set
bem examples/sphcap results onaxis --freq 1000 --variant fine  # variant filter
```

`bem MODEL export-web` writes an auralizer web bundle (consumed by the
bemkit-auralizer framework; the format contract is that repo's
docs/bundle-format.md, guarded there by a golden-manifest test). The
diagnostic `bem MODEL check ...` family ships separately via the
`bemkit.cli` entry-point group (private `bemlab` package).

`bem MODEL plot [TYPE] [SET]` options: `--variant` (repeatable), `--freq` (plane/sphere/contour),
`--freqs` (polar), `--angles` (freq/ir on a circular set; default `0,±30,±60,±90`),
`--ref` (onaxis/max/globalmax/number), `--clim LO HI`, `--contours DB` (0 disables),
`--linear-freq`, `--value-func`, `--filled`

`bem MODEL geometry` options: `--variant` (repeatable), `--no-bc`, `--no-mesh`, `--no-grid`,
`--mesh-opacity`, `--mesh-color`, `--grid-color`, `--point-size`, `--show-edges`,
`--bc-vibrating-color`, `--camera` (xz/xy/yz/iso), `--background`, `--screenshot`

### Normalization: raw / acc / auto

`acc` divides by `j·ρ₀·ω·A_vib`, so it only means anything for a **velocity-driven**
model. A scattering model (plane wave / point source against a rigid or absorbing
surface) has no vibrating area, and `SimulationData.A_vib` is None.

- `norm="raw"` — NumCalc's own output; always available; the `SimulationData` default.
- `norm="acc"` — raises `NoVelocityDrive` when there is no drive, rather than
  returning raw under an acc label (which is what it used to do).
- `norm="auto"` — acc when available, raw otherwise. The **viz default**, so plots
  label their axis from the resolved norm and never claim Pa/m/s² over Pascals.

`sim.is_velocity_driven` / `result.has_acc` expose the state;
`sim.effective_norm(norm)` / `result.effective_norm(norm)` resolve it. A scattering
`BEMResult` carries `pressure_acc = None` and stays fully usable as raw.

`A_vib` counts each element once and only `VELO` ranges with a non-zero amplitude
— `VELO 0.0` is how NumCalc spells sound-hard, and PRES/ADMI/IMPE are conditions,
not sources.

`bem MODEL results [SET]` (SET optional; default shows all fieldpoint sets) options: `--variant` (repeatable), `--quantity` (magnitude_db/magnitude/phase/real/imag),
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
sim = SimulationData.from_model("examples/sphcap", variant="fine")  # explicit variant

sim.set_names             # ['onaxis', 'polar', 'planeXZ', ...]
sim.frequencies           # (nf,) Hz array
sim.available_bins        # [1, 2, ..., 64]

# Fieldpoint data — dynamic attrs: sim.p_<set>, sim.v_<set>, sim.nodes_<set>.
# Raw (NumCalc's velocity-BC output) is the base everywhere; the acc
# normalization (assumes a velocity BC + A_vib) is opt-in via _acc_/norm="acc".
sim.p_onaxis              # (N, nf) complex — raw pressure (NumCalc output, default)
sim.p_acc_onaxis          # (N, nf) complex — normalized ÷ (j·rho0·ω·A_vib)
sim.v_onaxis              # (N, nf, 3) complex — raw 3D velocity (default)
sim.v_acc_onaxis          # (N, nf, 3) complex — normalized 3D velocity
sim.nodes_onaxis          # (N, 3) — coordinates in meters
# Or via methods:
sim.pressure("onaxis")              # raw (norm="raw" default)
sim.pressure("onaxis", norm="acc")  # normalized
sim.velocity("onaxis")              # raw 3D velocity (norm="raw" default)
sim.velocity("onaxis", norm="acc")  # normalized 3D velocity
sim.nodes("onaxis")                 # coordinates

# Boundary (surface mesh)
sim.boundary_pressure()             # (M, nf) complex — raw surface pressure (default)
sim.boundary_pressure(norm="acc")   # (M, nf) complex — normalized surface pressure
sim.boundary_velocity()             # (M, nf) complex — normal velocity (raw, default)
sim.boundary_velocity(norm="acc")   # (M, nf) complex — normalized normal velocity
sim.boundary_pressure_acc           # (M, nf) alias for boundary_pressure(norm="acc")
sim.boundary_velocity_acc           # (M, nf) alias for boundary_velocity(norm="acc")
sim.boundary_nodes                  # (M, 3)
sim.boundary_elements               # (T, 3) int — triangle connectivity

# Adapter for the viz functions (BEMResult is internal; carries raw + acc)
result = sim.result("onaxis")

# Single-bin access (no full load, for interactive viewers)
sim.pressure_at(12, domain="boundary")  # (M,) complex
sim.velocity_at(12, domain="eval")      # (N_total, 3) complex
```

## Configuration: config.toml

Per-model config in `examples/<name>/config.toml`. Solver settings live in the
required `[bem]` section; `modelname` is optional top-level (defaults to
directory name). Old configs with flat top-level solver keys fail with a
migration error. Unknown keys in known sections are rejected (typo guard) —
including the physics sections `[BOUNDARY]`, `[SYMMETRY]`, `["PLANE WAVES"]`,
`["POINT SOURCES"]` and the top-level `velocity` key, where a silent typo
used to mean plausible-but-wrong physics. Extra user tables (e.g. `[cad]`
data read by the model's own scripts) pass through untouched.
`ModelConfig.from_dict(raw, model_dir)` builds a validated config without a
file on disk (the seam for driving bemkit from another package); `[SYMMETRY]`
mirror planes are documented in HOWTO.md §4.

```toml
modelname = "sphcap"      # optional

[bem]
sr = 48000
nbins = 65                # n_frequencies = nbins - 1 (DC excluded)
air_density = 1.2
speed_of_sound = 340.0
bem_type = 4              # 0=TBEM O(N³), 1=SLFMBEM O(N^1.5), 4=DMLFMBEM O(N log N) [default]
time_factor = -1.0
is_internal = 0
# Solver knobs; defaults are NumCalc's own, so omitting them changes nothing.
solver = 0                # ISOLVER: 0 iterative (CGS), 4 direct (needs bem_type=0)
min_expansion_terms = 0   # 0 = NumCalc default (8); otherwise 5..10
cluster_edge_length = 0.0 # ClusEdgL0 (m); 0 = NumCalc default
preconditioner = 0        # IPRECOND in [-3, 2]; negative = incomplete-LU depth

[nc]                      # optional raw escape hatch: sets Deck fields directly
min_expansion_terms = 9   # applied last, wins over [bem]; scalars only

[geometry]                # optional: script that writes geometry/ sources
exec = "./make_geometry.py"   # string or list; run by `bem mesh`

[mesh]
lc_min = 5.0              # mm — ~6 elements per wavelength at max freq
lc_max = 25.0
dist_min = 50.0
dist_max = 100.0
distance = "euclidean"    # euclidean [default] | geodesic (STEP/CAD only)

[groups]                  # optional: map bemkit's BC role onto this mesh's
vibrating = "SourceDisc"  #   physical-group names (foreign meshers rarely use
                          #   "vibrating"). Everything outside it is rigid.

[fieldpoints.onaxis]      # declarative set: type + make_fields_* kwargs
type = "onaxis"           # onaxis | polar | plane
dim = "z"
radii = [0.1, 0.25, 0.5]  # meters

[fieldpoints.custom]      # pre-made file set
file = "fieldpoints/custom.npy"   # relative to model dir (or absolute)
```

### Mesh sizing distance metric

`distance` selects how the `lc_min`→`lc_max` ramp measures distance from the
vibrating surface; the four size knobs are unchanged by it.

- `euclidean` (default) — gmsh's own `Distance` field: straight line through
  space. Over-refines geometry that folds back on itself (the outer wall of a
  horn is Euclidean-near the piston but acoustically far).
- `geodesic` — distance measured *along* the surface
  (`bemkit/meshing/geodesic.py`: multi-source Dijkstra on the mesh edge graph).
  Gmsh has no geodesic field, so this runs a **two-pass** mesh inside
  `_mesh_from_step`: a coarse uniform pilot mesh carries the distance
  computation, is pushed back as a background view (`addListData` with `"ST"`
  scalar triangles — list data survives the `mesh.clear()` between passes,
  node-tagged model data would not), then the real graded mesh is generated.
  **STEP/CAD only** — `pipeline._generate_mesh` raises for scan sources, which
  have no vibrating group to measure from. Unreachable nodes (non-watertight
  model) are sized at `lc_max` with a loud warning. Needs scipy.

Do not suppress `Mesh.MeshSizeExtendFromBoundary`/`MeshSizeFromPoints` in the
geodesic pass: measured on `examples/sphcap`, doing so makes the far field
overshoot `lc_max` by ~50% (25 mm cap → 38 mm edges), because boundary-extended
sizes are what hold element growth in check across a curved face.

### Fieldpoint sets

Each `[fieldpoints.<name>]` table is one set with exactly one of:
- `type` — a generator (`onaxis`/`polar`/`plane`); remaining keys pass as
  kwargs to `make_fields_<type>` and points are generated in-memory at export
  (no `.npy`, no staleness)
- `file` — an `.npy` path

With no named sets, all `fieldpoints/*.npy` are loaded, one set per file. The
reserved `exec` key (`[fieldpoints] exec = "./make_fieldpoints.py"`) names a
script that runs at `bem export` before loading — the escape hatch for custom
point clouds (it typically writes `fieldpoints/*.npy`).

Generator reference (`bemkit.sim.fields`, also exported from the package root
for use in scripts):
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
`be.out/` (or packed `be.npz`) via `SimulationData`.

### Explicit variants — `[variants]` table

When `[variants]` is present, `bem` uses it instead of scanning `mesh/`. Each
entry's `mesh` key is either a **string** (trusted mesh-file stem, used as-is,
never remeshed — defaults to the variant name) or a **table** of `[mesh]`
parameter overrides (the variant's mesh is then *generated* as
`mesh/<variant>.msh` from a `geometry/` source: `source = "<stem>"`, else
`geometry/<variant>.*`, else the sole source).

Same mesh, different boundary conditions:

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

Same geometry, different mesh resolutions:

```toml
[mesh]                 # shared defaults
lc_min = 5.0
lc_max = 25.0

[variants.fine]
mesh = { lc_min = 2.5, lc_max = 12.0 }

[variants.coarse]
mesh = { lc_min = 10.0, lc_max = 50.0 }
```

Geometry sweeps: have the `[geometry]` exec script write several source files —
without `[variants]`, every `geometry/` source becomes its own variant
(mesh/<stem>.msh) automatically.

## Runner / Remote Execution

A variant directory `numcalc/{variant}/` is a self-contained, relocatable job bundle: NumCalc
runs with it as CWD using only relative paths, and writes results back into it. Frequency
bins are independent (`-istart`/`-iend`), and `be.out/be.{i}/` outputs are disjoint, so one
variant can be split across several processes/machines and merged trivially (no merge step
needed — outputs just land in the same `be.out/`).

- `Runner` ABC (`bemkit.numcalc.runner`): `submit(job)` / `poll(handle)` / `fetch(job)`.
  `LocalRunner` is the only implementation today (subprocess with CWD=variant dir); SSH or
  scheduler-based runners would implement the same interface — transfer the bundle, start
  NumCalc, report status, bring `be.out/` back.
- `Job(name, nc_dir, steps=(istart, iend) | None)` — one NumCalc invocation, optionally
  scoped to a 1-based inclusive bin range.
- `bem MODEL run` only computes frequency bins absent from `be.out/`/`be.npz` (the
  default) — it uses `bemkit.numcalc.status.missing_bins(nc_dir, nf)` to compute the gap, so
  partial/failed runs resume without redoing completed bins, and a complete run is a
  no-op. `--force` recomputes everything from scratch — required after re-exporting a
  changed mesh or config, since bin presence says nothing about which export produced it.
- `bem MODEL run` **detaches by default**: it re-execs `python -m bemkit.cli ... run
  --foreground` with `start_new_session=True` (so the run outlives the terminal) and
  returns. bemkit's own progress goes to `numcalc/run.log`; NumCalc's own output still
  goes to the per-variant `NC*.out`. `--foreground` runs inline and doubles as the
  supervisor's recursion guard. Everything decidable up front — NumCalc on PATH, a live
  run, an empty job list — is checked in `_detach_run` (`cli/_pipeline.py`) *before* the
  fork, so those failures reach the terminal rather than a log nobody is watching.
  `bem MODEL all` stays foreground: it is the blocking batch entry point.
- **A live NumCalc aborts `run`.** Two processes in one variant directory race each
  other's `be.out/` writes, so `running_variants(model_dir, variants)`
  (`bemkit.numcalc.status`) gates the run and it raises rather than quietly doing the other
  variants. `--force` is *not* an override — it `rmtree`s `be.out/`, which under a live
  solver is the one combination that must stay impossible. `--restart` kills the
  processes (SIGTERM, then SIGKILL) and then proceeds; on its own it resumes the missing
  bins, since a bin killed mid-write fails `complete_bins` and is recomputed anyway.
- `bem ps` lists every running NumCalc across *all* models (MODEL_DIR is ignored),
  one line per variant: model, variant, completed/total bins, process count, pids. It is
  the cross-model companion to `bem MODEL` status, needed once runs detach — the answer
  lives in each process's cwd (`<model>/numcalc/<variant>/`), read via
  `scan_numcalc()` (`bemkit.numcalc.status`), which is the undirected form of `numcalc_pids`.
- `bem MODEL pack` **deletes `be.out/` by default**; `--keep-original` opts out. The
  deletion is gated on `_verify_round_trip` (every bin and quantity re-read from ASCII
  and compared at `rtol=1e-5`, keeping `be.out/` on any mismatch) and is reversible via
  `unpack_variant`, which is what makes it safe as a default. `pack` skips a variant with
  a live NumCalc (skips, not aborts — the finished variants should still pack). The
  library-level `pack_variant(delete_original=False)` default is unchanged; only the
  pipeline stage and CLI default flipped.
- `bem MODEL run -j N` is the single load knob: N = max concurrent NumCalc
  processes, **default 1**. BEM peak memory, not CPU, is what limits a run —
  several concurrent DMLFMBEM solves on a large mesh exhaust RAM, and the
  heavy/light interleaving in `_queue_order` flattens that peak without
  bounding it. `plan_jobs(pairs, nf, max_jobs=N)` decides the rest — one job
  per variant, and when variants would leave slots idle, variants are fanned
  out into bin ranges (`-istart`/`-iend`) until N jobs exist. Slots go to the
  costliest variants, ranges are cut for equal cost (bin cost ∝ frequency
  index, variant cost ∝ element count), and the queue alternates heavy/light
  so concurrent jobs mix big and small rather than stacking memory peaks. On a
  resume (the default) the cuts apply to the *missing* bins and never span a
  bin that already has results, so a gap can yield more than N jobs. The cost
  model is a scheduling heuristic only — results never depend on it.
- **Partial-band runs.** `--freq HZ...`, `--bins N...` and `--preview N` on
  `run`/`all` restrict a run to a subset of bins; `bemkit.config.resolve_bins`
  snaps frequencies onto the model's grid (ties resolve low) and `plan_jobs`
  intersects the result with the bins it would otherwise schedule. Because the
  subset stays on the FFT grid, a later `bem MODEL run` fills in the rest and
  the results merge with no extra step. `--force` refuses to combine with a
  selection: it discards every bin, so the unnamed ones would come from the
  previous export. Sparse results are usable — frequency plots mark the
  computed bins — but anything crossing the whole band (`plot ir`, `plot etc`)
  raises `IncompleteBand` and is skipped with a reason.
- `bemkit.numcalc.status`: `available_bins(nc_dir)`, `missing_bins(nc_dir, nf)`,
  `is_complete(nc_dir, nf)` — derive job state directly from `be.out/`/`be.npz`, no separate
  bookkeeping file.
- `be.npz` shadows `be.out/` everywhere, so `bem MODEL run` removes a leftover `be.npz`
  for every variant it runs (on a resume its bins are first restored into `be.out/` so
  the runs merge); re-run `pack` afterwards. A `--force` recompute also deletes the
  variant's old `NC*.out` logs so superseded warnings don't linger in `bem status`;
  resumes keep them.
- Live jobs are detected, not tracked: `bemkit.numcalc.status.numcalc_pids(nc_dir)` scans
  /proc for NumCalc processes whose CWD is the variant dir (works for `bem run` and
  manual `cd numcalc/<v> && NumCalc`; returns [] on platforms without /proc).
  `bem status` shows them (with `current_step` from the newest NC*.out log), and
  `bem run` skips variants that are already running so two NumCalc processes never
  race the same be.out/.
- Bin presence ≠ convergence: NumCalc records non-convergence only in its log.
  `bemkit.numcalc.status.solver_warnings(nc_dir)` scans `NC*.out` for the two health strings
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

Emitted by `Deck.write` (`bemkit/numcalc/deck.py`); there is no template file.
Physical group IDs: `RIGID=1` (zero velocity, default), `VIBRATING=2` (specified velocity).
STEP colors: Blue (0,0,255) = rigid, Red (255,0,0) = vibrating.

### Driven-group resolution

`meshing/generate.py` names its groups `rigid`/`vibrating`, but foreign meshes
use their own (Akabak writes `HornWalls`/`SourceDisc`).
`resolve_vibrating_group(groups, name)` in `bemkit/meshing/io.py` is the single
lookup: it matches `name` from `[groups] vibrating` if given, else the default
`vibrating`, **case-insensitively**. It resolves by *name*, never by tag ID —
foreign meshes routinely invert bemkit's `RIGID=1`/`VIBRATING=2` numbering.

Failure modes are deliberately loud, because a surface with no driven elements
is a valid NumCalc job that returns silence:
- an explicitly configured name absent from the mesh **raises** at export;
- a mesh with physical groups where none resolved, and no `[BOUNDARY]` to
  compensate, prints a WARNING naming the available groups.

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

Condition types (`type`/`value` on a geometric entry; `velocity`/`v` stays the
VELO shorthand — give one or the other, never both):

```toml
[BOUNDARY]
soft     = { near = [0, 0.1, 0], type = "pres" }                 # 0 Pa = sound-soft
absorber = { near = [0, 0, -0.1], radius = 0.05, type = "admi", value = [0.4, -0.2] }
driver   = { near = [0, 0, 0.05], type = "velo", value = [1.0, 0.5] }  # complex
```

`type` ∈ velo/pres/admi/impe (`BC_TYPES` in config.py); `value` is a number or
`[re, im]`. **admi/impe are ρ₀·c-normalized**, so 1.0 is a matched termination —
values > 100 are rejected as almost certainly SI. TRAD/TRIM are refused by name
(NumCalc parses then hard-errors). `resolve_boundary` returns `BCRange(start, end,
kind, real, imag)`; literal `[s, e, v]` entries stay VELO-only.

Geometric anchors (`bemkit/sim/boundary.py`) — resolved against the mesh at
export time, so they survive remeshing; anchors/radii in meters:
```toml
[BOUNDARY.ear]
spot  = { near = [0.06, 0.0, 0.10] }                  # nearest element
patch = { near = [0.0, 0.0, 0.05], radius = 0.01 }    # all centroids within radius
multi = { near = [[x1,y1,z1], [x2,y2,z2]], velocity = 0.5 }  # one element per anchor
```
Export raises if an anchor is >5% of the mesh extent from its nearest element,
or a `radius` selects no elements (units/frame mistake). A variant name is
matched against `[BOUNDARY.<name>]` first with its full name, then with the
`__recipe`/`@profile` suffix stripped (`ear__r05` → `[BOUNDARY.ear]`).

### PLANE WAVES / POINT SOURCES (incident fields)

`["PLANE WAVES"]` / `["POINT SOURCES"]` tables in config.toml pass through to
NC.inp as 8-value rows (malformed rows raise at export — header counts must
match emitted lines). The two sections share a layout but **not** a meaning for
fields 1-3:

- `["PLANE WAVES"]` → `[nr, dx, dy, dz, ampl_real, -1, ampl_imag, -1]` — a
  travel **direction**, normalized by NumCalc itself (`NC_Input.cpp:1024`), so
  it need not be a unit vector but must not be all zeros.
- `["POINT SOURCES"]` → `[nr, x0, y0, z0, ampl_real, -1, ampl_imag, -1]` — a
  **position** in meters.

A row copied between the sections is therefore accepted and silently wrong;
`_SOURCE_TRIPLE` in `sim/builder.py` keeps the error message honest about which
triple the section wants. Keep point sources ≥1 average edge length off the
surface (arXiv:2312.00005, Hint 3) — the Green's function is singular at zero
distance.

Curve refs are always written as `-1`: frequency-dependent amplitudes (CURVES)
are not supported. NumCalc requires curve refs to be all `-1` or all non-`-1`
within one line, for sources and boundary conditions alike
(`NC_Input.cpp:1244`). Results default to NumCalc's raw output (the natural
reading for scattering); the acc normalization, which assumes a velocity BC,
is opt-in via `norm="acc"` / `p_acc_<set>`.

## Analytic Solution

`bemkit.analysis.analytic` (exported as `sphcap_pressure` from the package root) computes
exact pressure for axisymmetric spherical cap vibration via SH expansion.
`sphcap_pressure(eval_points, frequencies, c0, rho0, angle_deg, radius_mm, nmax)` → raw pressure array.
Normalization matches BEM pipeline: velocity BC (v=1) → raw pressure → ÷(rho0·omega·A_vib) → pressure_acc.

## Visualization: bemkit.viz

Composable matplotlib/PyVista visualizations. All 2D functions take an Axes and BEMResult, return artists.

2D plots: `freq_magnitude`, `freq_phase`, `freq_real_imag`, `impulse_response`, `energy_time_curve`, `polar_directivity`, `polar_balloon_slice`, `heatmap`, `contour`, `scatter_field`, `spherical_heatmap`

### Dataset-tailored plots

These read the *geometry* of a fieldpoint set and refuse the wrong one, so
`bem MODEL plot <type>` over all sets plots the ones that fit and skips the
rest with a reason. `bemkit/viz/_angles.py` answers the two questions they
share — is this set a circle about the origin (`circle_angles`, and the angle
of every point), is it collinear through the origin (`axis_positions`) — from
the coordinates alone, since a `BEMResult` carries no record of the
`[fieldpoints.<name>]` table that made it.

| Function | Set | What it shows |
|---|---|---|
| `isobaric` (`viz.directivity`) | circular | angle × frequency map, level in color |
| `angle_response` (`viz.freq`) | circular | frequency response at 0°, ±30°, ±60°, ±90° |
| `angle_impulse_response` (`viz.time`) | circular | IRs of selected directions, overlaid |
| `onaxis_response` (`viz.freq`) | on-axis | frequency response × r (1/r decay removed) |

Angle convention (shared by `polar_directivity(angle_key="auto")`): for a
circle whose plane normal is axis *n*, 0° is on axis `(n+1)%3` and +90° on
`(n+2)%3`. A `dims = "zx"` polar set therefore has 0° on +z. Angles run
-180°…180°, and `angle_indices` snaps a requested direction to the nearest
measured one, dropping requests outside the covered arc — so a half circle
just yields fewer curves.

dB reference (`directivity_levels`, used by `isobaric`, `polar_directivity`
and `angle_response`): `"onaxis"` (default for the maps — every frequency
referenced to its own 0°, so the plot is directivity only), `"max"`,
`"globalmax"`, or a number for absolute levels. `--ref` on the CLI.

A ±θ pair always shares a color and is separated by line style
(`_style.angle_styles`), so the pairing survives grayscale and color blindness
— on an axisymmetric source the two curves land on top of each other, which is
itself the symmetry check.

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
