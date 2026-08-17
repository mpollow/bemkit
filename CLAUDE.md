# bemkit

Public, GPL-3.0-licensed repo. Python library and `bem` CLI for Mesh2HRTF NumCalc
simulations, driven from a Python `model.py` rather than a config grammar.

## Project Structure

A monorepo of **two distributions**. Each installs on its own.

```
packages/
├── numcalckit/          the NumCalc contract, numpy only
│   └── numcalckit/      #   deck.py      — Deck: NC.inp as one object, read/write
│                       #   meshfiles.py — ObjectMesh/EvalGrid writers (arrays in)
│                       #   readers.py   — be.out parsers, load_bins
│                       #   packing.py   — be.out/ <-> be.npz
│                       #   runner.py    — Job/Runner/plan_jobs/solver_flags
│                       #   status.py    — bins, convergence, warnings, /proc scan
└── bemkit/             modelling + solving + results + 3D inspection
    └── bemkit/         #   model/     — Mesh, Model, selectors, conditions, sources,
                        #                field generators (numpy only, no I/O)
                        #   geometry/  — load_mesh, gmsh generation, mmg, quality  [gmsh]
                        #   solvers/   — Solver protocol, NumCalcSolver, BemppSolver
                        #   results/   — Results, stores, acc normalization
                        #   view/      — BEMScene / ResultsViewer (3D),
                        #                convergence plots, web export  [view]
                        #   cli/       — the `bem` entry point
                        #   study.py   — model.py / params.toml / runs/ layout
```

One spine, and everything hangs off it:

```
Mesh  →  Model(mesh, frequencies, medium, bcs, sources, fields)  →  Solver  →  Results
```

`Model` is pure data: numpy only, no I/O, no solver knowledge. That is what
lets two solvers be cross-validated on one problem, and what makes every layer
above it unit-testable without a mesh file or a solver binary.

Install for development — one command, both editable packages:

```bash
pip install -e packages/numcalckit -e packages/bemkit
pytest                     # runs both suites (testpaths in pyproject.toml)
```

Optional extras on `bemkit`: `view` (pyvista + matplotlib, for `geometry`/
`results`/`export-web`), `meshing` (gmsh + scipy, for mesh generation),
`analytic` (scipy, for `sphcap_pressure`), `all` (everything). `BemppSolver`
needs `bempp-cl`, installed separately — it is imported lazily inside
`solve()`, so importing `bemkit` or `bemkit.solvers.bempp` never requires it.

**Import direction is one-way and enforced** by
`packages/bemkit/tests/test_boundaries.py`:

- `model/` imports nothing of ours (its one scipy user, `analytic.py`, is
  reached through a lazy `__getattr__` so the rule holds for the eager path).
- `geometry/`, `solvers/`, `results/` — the *core* — import `model/` and each
  other, but never `view/`, and never pyvista/matplotlib.
- `view/`, `cli/` may import anything, but `cli/__init__.py` only imports
  `view` lazily inside functions, so `bem --help` never pays for pyvista.
- `numcalckit` imports nothing from `bemkit` and needs only numpy — checked on
  both the library and its own test suite, which is what keeps it installable
  on its own.

`pip install -e .` installs the `bem` console script (entry point
`bemkit.cli:main`). Without installing, `python -m bemkit.cli` is identical.
`bem` also loads third-party subcommands via the `bemkit.cli` entry-point
group (what the private `bemlab` package uses for its own verbs, e.g. `bem
STUDY verify`); a plugin that redefines a name bemkit already owns is skipped
with a warning.

## The model layer (`bemkit.model`)

```python
Mesh(vertices, triangles, groups)          # (N,3) float, (M,3) int, {name: element idx}
Medium(c=343.0, rho=1.2)
Model(mesh, frequencies, medium=None, *, bcs=(), sources=(), fields=None,
      exterior=True, time_factor=-1.0)
```

`Mesh` is **always in metres** — the `.msh`-is-millimetres / `.stl`-is-metres
split is a `bemkit.geometry.load_mesh` concern and stops being a live trap
anywhere else. `frequencies` is an explicit ascending Hz array; there is no
`sr`/`nbins` concept in the model itself — `fft_grid(sr, nbins)` is one helper
that returns such an array, needed only when a time-domain result is wanted
(`Model.is_uniform_band` gates that).

`time_factor` accepts only exactly `-1.0` (e^-iωt, the default) or `+1.0`:
NumCalc coerces anything else to `+1.0` and flips the sign of every imaginary
part (`NC_Input.cpp:334`), so there is no safe default to fall back on for a
typo.

### Boundary conditions: selectors × conditions

*Where* a condition applies and *what* it is are orthogonal, which is what
keeps four selectors times four conditions from needing a grammar key per
combination:

```python
from bemkit.model import BC, Group, Near, Where, Velocity, Pressure, Admittance

BC(Group("vibrating"),             Velocity(1.0))
BC(Near((0, 0, 0.05), radius=0.02), Velocity(1.0 + 0.5j))
BC(Group("wall"),                  Admittance(felt))          # array over frequencies
BC(Where(lambda c: c[:, 2] < 0),   Pressure(0))                # predicate on centroids
```

Selectors (`bemkit.model.select`) resolve against a `Mesh` to element indices:

- `Group(name)` — a mesh physical group, matched **case-insensitively** by
  name (never by tag id — foreign meshers, e.g. Akabak's `SourceDisc`,
  routinely invert bemkit's own `RIGID=1`/`VIBRATING=2` numbering). Raises,
  naming the groups that *are* there, on a miss.
- `Near(point, radius=None)` — nearest element to `point` (metres), or every
  element within `radius`. An anchor further than 5% of the mesh extent from
  its nearest element, or a `radius` that selects nothing, raises — both catch
  a wrong unit or coordinate frame. Anchors resolve at solve time, so they
  survive a remesh.
- `Elements(indices)` — literal 0-based indices; tied to one triangulation.
- `Where(predicate)` — a `(M, 3) centroids -> (M,) bool` callable; not
  serialisable, which is why a run snapshots the *resolved* elements rather
  than the selector.

Conditions (`bemkit.model.conditions`) carry the value, and NumCalc's element
line: `Velocity` (`VELO`, `0` = sound-hard, the default for any unselected
element), `Pressure` (`PRES`, `0` = sound-soft), `Admittance`/`Impedance`
(`ADMI`/`IMPE`, **ρ₀·c-normalised** — `1.0` is a matched, anechoic
termination; a magnitude above 100 is rejected as almost certainly SI).
`TRAD`/`TRIM` are not offered: NumCalc parses them and then hard-errors.

**Frequency dependence is Admittance/Impedance-only**, and that is physics,
not an arbitrary restriction. The BEM is linear, so a frequency-dependent
*excitation* is a scale factor on the answer — `Velocity`/`Pressure` and every
source therefore take a scalar (possibly complex) value, and passing an array
raises, naming `Results.scaled_by` as the alternative: solve once at unit
amplitude, then

```python
res = solver.solve(model)
spk = res.scaled_by(driver_response)   # (nf,) complex — no re-solve
```

Admittance/impedance are different in kind — they enter the boundary
*operator*, so the system matrix genuinely changes per frequency — and accept
either an `(nf,)` array or a callable `f_hz -> value`. This is exactly why
NumCalc's `CURVES` exist for `ADMI` and nothing else (see NC.inp Format,
below).

### Sources and field points

`PlaneWave(direction, amplitude=1.0)` / `PointSource(position, amplitude=1.0)`
— same three numbers, different meaning (a travel direction vs. a metres
position), which is why they are separate types rather than one with a mode
flag. Keep a point source at least one average element edge length off the
surface (arXiv:2312.00005, Hint 3) — `bem STUDY check` verifies this.

`Model.fields` is a plain `{name: (N, 3) array}` dict, metres. Generators
(`bemkit.model.fields`) return such arrays: `on_axis(axis, distances)`,
`circle(dims, res, radius)`, `plane(dims, res, limits)`; anything else is
`np.load(...)` or ordinary numpy composition. Validated against the mesh at
solve time (points inside the surface for an exterior problem, or too close to
it, are dropped) — the Möller-Trumbore/Eberly kernels carried over unchanged
in `bemkit.solvers.fieldpoints`.

### Vibrating area and acc normalisation

`Model.vibrating_area` sums the area of every element driven by a non-zero
`Velocity` condition (each element once, even if two selectors overlap;
`Velocity(0)` is sound-hard and contributes nothing) — `None` when nothing
drives the model, e.g. a scattering problem. It is what
`bemkit.results.acceleration_scale(model)` divides by; see Results, below.

## Geometry (`bemkit.geometry`)

`load_mesh(path, scale=None)` reads any mesh file into a `Mesh`, in metres.
Unit conversion happens here and nowhere else: `SCALE_BY_SUFFIX` maps `.msh`
(millimetres, GMSH's convention) to `0.001` and `.stl`/`.ply`/`.obj`/`.mesh`
(already metres) to `1.0`; pass `scale=` explicitly for anything unusual.
Physical groups come from the file's `gmsh:physical`/`field_data`, when
present.

`mesh_from_step(step_file, output_file, sizing, *, configure=None)` and
`mesh_from_surface(surface_file, output_file, sizing, *, configure=None)`
generate a graded GMSH mesh from CAD or a tessellated scan (`.ply`/`.stl`/
`.obj`, treated as metres and scaled to mm for GMSH). STEP surfaces are
classified by face colour — **blue (0,0,255) = rigid, red (255,0,0) =
vibrating** — and the vibrating tags are handed to `sizing`; a surface with no
colour tags is meshed as a single rigid group with a printed warning. There is
no CLI verb for this — a study's own script (or a one-off line in `model.py`)
calls it, typically once, ahead of `build()` reading the `.msh` back with
`load_mesh`.

`mesh_sphere_cap(output_file, sizing, *, radius_mm=50.0, angle_deg=30.0,
configure=None)` is the third generator and the only parametric one: a rigid
sphere with a vibrating polar cap, built from two revolved arcs (not a boolean
cut — the cap edge is then a *shared* curve of the two faces, which is what
`edge_field` measures from) and grouped `vibrating`/`rigid`. It exists because
`bemkit.model.analytic.sphcap_pressure` gives that shape an exact answer, so
sweeping its *discretisation* is a verification rather than a comparison of two
guesses — `models/sphcap_meshing`. It checks its own output before returning:
areas against the closed forms, plus watertightness, outward normals and
vertices on the sphere read back from the written file.

`sizing` builders (`bemkit.geometry.sizing`) install a background size field:

- `uniform_field(h)` — constant element size.
- `distance_field(lc_min, lc_max, dist_min, dist_max)` — GMSH's own
  `Distance` field: straight-line distance from the vibrating surface, ramped
  linearly between `dist_min`/`dist_max`. Over-refines geometry that folds
  back on itself (the outer wall of a horn is Euclidean-near the piston but
  acoustically far).
- `geodesic_field(lc_min, lc_max, dist_min, dist_max)` — distance measured
  *along* the surface (`bemkit.geometry.geodesic`: multi-source Dijkstra on
  the mesh edge graph, needs scipy). GMSH has no geodesic field, so
  `mesh_from_step` runs a coarse pilot pass to compute it, pushes it back as a
  background view, then generates the real graded mesh. **STEP/CAD only** —
  a scan has no vibrating group to measure from. Unreachable nodes (a
  non-watertight model) are sized at `lc_max` with a loud warning. Do not
  suppress `Mesh.MeshSizeExtendFromBoundary`/`MeshSizeFromPoints` in this
  pass: measured on `models/sphcap`, doing so overshoots `lc_max` by ~50%.
- `edge_field(lc_min, lc_max, dist_min, dist_max, *, curves=None,
  sampling=200)` — distance from *curves* rather than faces, so the refinement
  is a band around a one-dimensional feature: the rim of a driven cap (where
  the velocity BC jumps and the faceted rim under-states the true circle), or a
  sharp geometric edge. `curves=None` resolves to `patch_edge_curves(gmsh,
  sources)` — the curves where the driven faces meet *other* faces, which is
  narrower than `getBoundary` and deliberately so: a surface of revolution's
  boundary also holds its seam and a degenerate curve at each pole, and
  measuring from those puts a fine meridian stripe and a fine pole into a mesh
  that asked for a fine rim. Unlike the other builders this one **does** set
  `Mesh.MeshSizeExtendFromBoundary`/`MeshSizeFromPoints` to 0, because gmsh
  would otherwise spread `lc_min` from those very curves across the whole
  surface — measured on a 30° cap, a (1.33, 5.31) ramp left the far side at
  3.5 mm instead of 5.3 and more than doubled the element count. `configure`
  still runs afterwards and can put it back.

`bemkit.geometry.mmg.curvature_sizes(mesh, hmin, hmax)` / `write_sol(path,
sizes)` — per-vertex isotropic sizing from face-normal curvature, for `mmgs`
remeshing (ported from the private `bemlab`'s `metric.py`).

`bemkit.geometry.quality.check_mesh(vertices, triangles, *, max_frequency,
speed_of_sound, sources=())` runs the diagnostics NumCalc itself does not:
watertightness, normal orientation, elements per wavelength, element
regularity, neighbour size uniformity, point-source clearance. Covers hints
1/2/3/4/6 of arXiv:2312.00005; pure numpy, no mesher needed. `check_watertight`
underlies both watertightness and orientation. Severities: `error` (the result
will be wrong), `warning` (likely degraded accuracy), `ok` (checked and fine).
Driven via `bem STUDY check` (below).

**`elements_per_wavelength` misjudges a graded mesh**, and knowingly: it
measures the globally longest edge, so a mesh that puts its coarse elements
where the field is weak fails on the strength of its cheapest region. Measured
in `models/sphcap_meshing`, it rejects the two meshes that are the *most*
accurate per element in the whole study (1.2 el/λ reported; 3× fewer elements
than the uniform mesh of equal accuracy). It is right about every uniform mesh
and has no way to know which group is driven — a per-group version would fix
it, and does not exist yet.

Two other checks are worth knowing about. **Orientation** answers a question
NumCalc's own `-check_normals` does not: that flag only verifies normals are
*consistent*, so a uniformly inverted mesh passes it — `check_mesh` adds the
signed-volume (divergence theorem) test for *outward*, which the paper calls
the most common error that still produces plausible output. **Element
regularity** uses a normalised aspect ratio (`longest_edge / (2·inradius)`,
1.0 = equilateral), because the naive longest/shortest-edge ratio scores a
degenerate needle at a harmless-looking 2.

## Solvers (`bemkit.solvers`)

```python
class Solver(Protocol):
    name: str
    def solve(self, model, *, workdir=None, bins=None, jobs=1) -> Results: ...
    def status(self, workdir) -> Status: ...
```

`workdir` reconciles two lifecycles. NumCalc needs one — it is a binary that
reads `./NC.inp` from its cwd, and the directory *is* the resumable,
inspectable, relocatable bundle. With `workdir=None` a batch backend uses a
temporary directory and materialises `Results` into memory before cleanup —
right for a quick comparison, wrong for a model too big to hold in RAM.
`bins`/`jobs` are meaningful to batch backends and no-ops for in-process ones.

`NumCalcSolver` (`name = "numcalc"`) is the write-run-ingest pipeline over
`model_to_deck` (the sole place NumCalc-isms live: `bemkit.model` above knows
nothing of decks or element ranges, `numcalckit` below knows nothing of
models). `.export(model, workdir)` writes `ObjectMesh/`, `EvalGrid/`, `NC.inp`
and `eval_offsets.json` without solving — used on its own by `bem STUDY
estimate-ram` and by the 3D viewers, which need a bundle to draw but not a
finished run. `.solve(...)` additionally takes `options=` (NumCalc argv flags
— see `numcalckit.solver_flags`, and the CLI section below), and raises
`RuntimeError` naming the failed jobs' logs, or `ValueError` when every field
point was dropped by validation.

**Solver knobs are not yet exposed on `Model`.** `bem_type`, `isolver`
(`ISOLVER`), `cluster_edge_length` (`ClusEdgL0`), `preconditioner`
(`IPRECOND`) and `min_expansion_terms` live on `numcalckit.deck.Deck` with
NumCalc's own defaults and `Deck.validate`'s guard against the two
combinations NumCalc accepts and silently downgrades (`isolver=4` with an FMM
`bem_type`; `preconditioner` alongside `isolver=4`) — but nothing on the
`Model` → `NumCalcSolver` path sets them, and there is no successor to the old
`[nc]` escape hatch. Tuning one today means calling
`bemkit.solvers.numcalc.model_to_deck(model, n_eval_nodes=..., n_eval_elements=...)`
directly, setting the field, and driving `Deck.write`/`numcalckit.run_jobs`
by hand rather than through `NumCalcSolver.solve`.

`BemppSolver` (`name = "bempp"`) is a second, in-process backend
(`bempp_cl`, imported lazily inside `solve()`) that exists to *cross-check*
NumCalc, not replace it — see `packages/bemkit/tests/solvers/test_cross_validation.py`,
which solves `models/sphcap` on both and compares each to
`bemkit.model.analytic.sphcap_pressure`. It refuses, naming the feature,
anything outside a deliberately small subset: **exterior** problems only,
**uniform `Velocity`** conditions only (no `Pressure`/`Admittance`/`Impedance`,
no frequency dependence), **no incident sources**. Formulation: direct
(non-regularised) exterior Neumann BIE, Galerkin, piecewise-constant on both
traces — irregular frequencies exist (lowest for a sphere of radius `a` is
`ka = π`) and are avoided by staying below them, not regularised away. **This
is where the two backends differ**: NumCalc does *not* have that limitation, so
a cross-validation band has to stay inside bempp's, not NumCalc's.

**NumCalc is Burton-Miller.** `NC_EquationSystem.cpp:85` sets the coupling
constant `zBta3 = i·τ/k` for an exterior problem and `0` for an interior one,
which is the combined-formulation coupling — so exterior sweeps do not have to
dodge the interior Dirichlet eigenvalues. That is measured, not merely read:
`models/sphcap_meshing` sweeps a 50 mm sphere over ka = 0.23 … 14.8, crossing
dozens of them, and `compare_analytic.py`'s spike check finds no isolated error
peak on any of its nine meshes.

**Time convention is measured, not assumed**: bempp's kernel is an outgoing
wave under `e^{-iωt}`, the opposite of NumCalc's `time_factor=-1.0` default,
so `BemppSolver` conjugates its result for `time_factor=-1.0` and leaves it
alone for `+1.0`.

**Mirror-plane symmetry (`SYMMETRY`) has not been ported.** `Model` has no
symmetry field, and `model_to_deck` never writes a `SYMMETRY` block — a study
that needs NumCalc's element-mirroring cannot express it from `model.py`
today (the pre-restructure `[SYMMETRY]` config grammar covered this; see git
history before `93b7ab7` for the mechanism). `bemkit.view` still reads and
draws whatever symmetry an `NC.inp` happens to carry (relevant to a bundle
built by hand or by older tooling), but nothing on the current spine writes
one.

## Results (`bemkit.results`)

`Results(store, frequencies, offsets, scale=None)` is solver-neutral access to
a solved `Model` — a NumCalc bundle and an in-process backend both end up
here. `store` is one of `MemoryStore` (arrays already in RAM),
`NpzStore` (lazy, cached `.npz`), fed by `.pressure(name=None)` /
`.velocity(name=None)` (eval-grid, split by field-point set) /
`.boundary_pressure()` / `.boundary_velocity()`.

`.scaled_by(spectrum)` returns a *view* multiplied by an `(nf,)` spectrum,
applied lazily on each read — the mechanism the linearity rule above relies
on, and how normalisation is expressed:

```python
from bemkit.results import acceleration_scale

acc = res.scaled_by(acceleration_scale(model))   # raises NoVelocityDrive if none
```

`acceleration_scale(model)` returns `1 / (j·ρ₀·ω·A_vib)` — NumCalc solves a
unit *velocity* BC and reports pressure for that drive; dividing by the
vibrating area re-expresses it per unit volume acceleration. It **raises**
`NoVelocityDrive` rather than silently returning raw under an acc label when
`model.vibrating_area` is `None` (a scattering model), because raw Pascals
mislabelled acc are wrong by a frequency-dependent factor and look entirely
plausible on a plot.

`.materialise()` applies the current scale eagerly and drops it, in place —
turns every later read into a pure dict lookup. `.save(path)` /
`Results.load(path)` round-trip through `.npz` (the *effective*, scaled
quantities); `path` may be a directory, in which case `results.npz` is used.
`Results.from_beout(workdir, frequencies, offsets)` ingests a NumCalc
`be.out/`/`be.npz` (missing bins come back `NaN`, distinct from a bin that
solved to an actual zero) and writes `results.npz` alongside so the ASCII
parse only happens once.

## The study contract (`bemkit.study`)

```
models/<name>/
├── model.py            required — build() -> Model, or variants() -> {name: Model}
├── params.toml          optional — flat scalars, read with tomllib, NOT validated
├── geometry/, mesh/      whatever build() needs to find its input
└── runs/<solver>/<variant>/    generated: one bundle per solver per variant
```

`model.py` must define `build()` (optionally `build(params)`, when
`params.toml` exists — the loader inspects the signature) or `variants() ->
{name: Model}`, which wins over `build()` when present. Variants are therefore
ordinary Python — a loop, a comprehension — not a `[variants]` table:

```python
def variants():
    base = dict(sr=32000, nbins=65, c=340.0, rho=1.2)
    return {name: build({**base, "mesh": name})
            for name in ("coarse", "standard", "fine")}
```

`params.toml` is deliberately unvalidated: a typo surfaces as a `KeyError` (or
whatever `build` itself raises) at build time, which is early enough for
research code — the trade for deleting 863 lines of TOML schema validation.
`load_study(path)` returns `{variant: Model}`; `bemkit.cli`'s `load()` wraps
it with a CLI-friendly message on a missing `model.py`. Anything `build()`
itself raises propagates with its own traceback — a study is Python, and a
`KeyError` inside it is a line number its author needs.

`workdir(study_dir, variant, solver="numcalc")` is `STUDY/runs/<solver>/
<variant>/` — self-contained and relocatable, since NumCalc runs with it as
CWD using only relative paths. Solver-in-the-path is what makes cross-
validation ordinary: `NumCalcSolver` and `BemppSolver` write side by side
without colliding. `match_variants(names, prefixes)` implements the CLI's
`--variant` prefix filter, repeatable and space-separated.

### The studies in `models/`

- `sphcap` — a 20° cap on a 50 mm sphere, three meshes from a committed STEP.
  **A fixture, not a research study**: `tests/solvers/test_cross_validation.py`
  pins its cap angle and its measured cap-area deficit, `tests/cli/test_check.py`
  pins `coarse`'s exact node and element counts, and `tests/geometry/test_load.py`
  reads `mesh/coarse.msh`. Change it and four test files move with it.
- `sphcap_meshing` — the same shape at 30°, nine meshes, asking which *meshing
  strategy* buys accuracy against the closed form. Parametric geometry
  (`mesh_sphere_cap`), no STEP. `README.md` there carries the measured result.
- `plate` — geodesic vs euclidean sizing, the one geometry where the two
  genuinely disagree (`sphcap_meshing` is the convex control where they must not).
- `hutubs21_glassesearbuds` — the real HRTF workload.

## Usage: the `bem` CLI

`STUDY_DIR` defaults to `.` when the first token is a known subcommand or
absent, so `bem solve` from inside a study directory is `bem . solve`; a
study directory literally named the same as a subcommand needs a `./` prefix.
With no subcommand, `bem STUDY` is `bem STUDY status`.

```bash
bem models/sphcap                       # status: per-variant export/solve state,
                                        #   live NumCalc jobs (⟳, current bin via /proc)
bem models/sphcap check                 # mesh + setup diagnostics (see Geometry, above)
bem models/sphcap check --strict        # exit non-zero on warnings too

bem models/sphcap solve                 # missing bins only; starts in the background
bem models/sphcap solve -j 4            # up to 4 concurrent NumCalc processes
bem models/sphcap solve --force         # recompute everything (mutually exclusive
                                        #   with a band selector — it would leave the
                                        #   unnamed bins from the previous model)
bem models/sphcap solve --foreground    # wait instead of detaching
bem models/sphcap solve --restart       # kill a running NumCalc first, then resume
bem models/sphcap solve --preview 8     # 8 bins log-spaced over the band
bem models/sphcap solve --freq 1000 4000    # nearest bins to these frequencies
bem models/sphcap solve --bins 4 16 64      # literal 1-based bins
bem models/sphcap solve --nitermax 5000     # max CGS iterations
bem models/sphcap solve --check-normals     # NumCalc verifies normal consistency
bem models/sphcap solve --adapt-fmmlength   # adapt multipole truncation to clusters

bem models/sphcap estimate-ram          # clustering only; RAM at the top bin
bem models/sphcap estimate-ram --all-bins

bem models/sphcap pack                  # be.out/ -> be.npz (~4x shrink), be.out/ removed
bem models/sphcap pack --keep-original

bem models/sphcap convergence           # solver quality per bin [view extra]
bem models/sphcap convergence --no-plot     # the summary table alone, no matplotlib
bem models/sphcap convergence --save q.png

bem ps                                  # every running NumCalc, any study (ignores STUDY_DIR)

# --variant filters by prefix, repeatable, goes AFTER the subcommand:
bem models/sphcap solve --variant fine
bem models/sphcap results onaxis --freq 1000 --variant fine

# 3D pre-solve inspection [view extra] — exports any bundle that is missing first
bem models/sphcap geometry
# 3D post-solve inspection [view extra]
bem models/sphcap results --freq 1000
bem models/sphcap results onaxis --freq 1000
```

There is no `mesh`/`export`/`init`/`all` verb any more — a study is Python, so
meshing is a script or a line in `model.py` (see Geometry, above), and
`solve()` exports the bundle it needs as its first step. `bem STUDY geometry`/
`results` do the same when a bundle is missing (`bemkit.view.bundle.
prepared_bundles`) — there being no separate export step means "inspect before
solving" does not mean "solve first".

**Resume is the default.** `solve` computes only the bins absent from
`be.out/`/`be.npz` (`numcalckit.status.missing_bins`), so an interrupted run
continues and a complete one is a no-op; `--force` discards everything first
(needed after `model.py` changes the mesh or the band — bin presence alone
says nothing about which model produced them). A resume also **refuses** if
the bins on disk were produced by a *different* model.py than the one loaded
now (`bem/cli/_commands.py::_model_digest`/`_guard_unchanged_model` fingerprint
the mesh, band, medium, resolved BCs, sources and field sets into
`runs/.../model.sha256`) — mixing two models' results in one `be.out/` would
otherwise pass silently.

`solve` **detaches by default**: it re-execs itself with `--foreground` and
`start_new_session=True`, and returns immediately. Progress goes to
`runs/run.log`; NumCalc's own output stays in the per-variant `NC*.out`.
Everything decidable up front (NumCalc on PATH, a live run, nothing left to
do) is checked before the detach, so those failures reach the terminal.

**A live NumCalc blocks a resume**, because two processes in one variant
directory race each other's `be.out/` writes; `--restart` kills it first
(SIGTERM, then SIGKILL) and resumes. `-j N` (default 1) is the only load
knob — BEM peak memory, not CPU, is what limits a run, and `plan_jobs` fans
missing bins out into ranges to fill N slots, heaviest variant first,
alternating heavy/light so concurrent jobs do not stack memory peaks.

`bem ps` lists every NumCalc process on the machine (not just this study),
reading model/variant/progress out of each one's cwd via a `/proc` scan — the
cross-study companion needed once runs detach.

`pack` deletes `be.out/` by default, gated on a full round-trip verification
at `rtol=1e-5` (kept on any mismatch); `--keep-original` opts out.
`estimate-ram` runs NumCalc's clustering pass only (`-estimate_ram`, no BEM
solve) at the top bin by default, where FMM memory peaks; it slightly
under-estimates (FMM matrices only, not mesh/result arrays).

Bin presence ≠ convergence: `numcalckit.status.solver_warnings` scans
`NC*.out` for NumCalc's two non-convergence strings and both `status`/`solve`
surface them — usually a defective mesh, fixed by re-meshing rather than
raising `--nitermax`.

`convergence` is the quantity behind that bit. `numcalckit.status.convergence`
reads the summary line NumCalc writes once per step (`CGS solver: number of
iterations = N, relative error = E`, `NC_CommonFunctions.cpp:1178`) into
`BinRecord`s — iterations, final residual, the per-step assemble/solve/post
seconds, and any warning, per bin. Two constants come from the same source and
are what `.converged` and the plot's reference lines are measured against:
`CGS_TOLERANCE = 1e-9` (`ErroIterSols`, `:39`) and `NITER_MAX_DEFAULT = 250`
(`niter_max_`, `NC_Main.cpp:83`) — note the solver stops at **1e-9**, not the
1e-6 the comment at `deck.py:243` implies. There is **no per-iteration residual
trace** in NumCalc's logs, so a residual-vs-iteration curve is not available;
frequency is the only x-axis these plots can have.

**Read those constants only from the tree the binary was built from**
(`projects_EXT/Mesh2HRTF/mesh2hrtf/NumCalc/src/`, per `solve_chain.sh`'s PATH).
A second checkout on this machine, `projects_EXT/mesh2hrtf-git-sf/…/Source/`,
still carries an older `int niter_max = 1500` — a line that is *commented out*
in the real tree and superseded by `niter_max_ = 250`. Taking the 1500
overstates the headroom by 6x and makes a run that nearly truncated look safe.
250 is low enough to bite: `models/hutubs21_glassesearbuds` passes
`--nitermax 4000` as a correctness requirement, because at the default eight
glasses bins stopped at residuals of 3.8e-6 to 9.4e-4.

Resumes leave overlapping logs in one variant directory (bins 1-45, then 34-45,
then 47-64…), so records are merged newest-first, and **a newer record without
a solve never displaces an older one that has one** — otherwise a run still in
flight, or killed after printing its step header, would blank out the bin the
results on disk actually came from.

`bem STUDY convergence` (`bemkit.view.convergence`, [view extra]) draws three
panels over one log-frequency axis: iterations, final residual against the
tolerance line, and seconds split into assemble/solve/post. Variants are an
**envelope** — min-max band, median, thin grey traces — not one colour each: a
study routinely has more variants than any categorical palette has slots, so
only outliers and non-converged bins are coloured and named. `--no-plot` prints
the summary table alone and needs nothing but `numcalckit`.

## Unit Conventions

| File Type | Units |
|-----------|-------|
| Field-point arrays (`Model.fields`, `.npy`) | **METERS** |
| Mesh `.msh` (GMSH) | **MILLIMETERS** |
| Mesh `.stl`/`.ply`/`.obj`/`.mesh` | **METERS** |
| `Mesh` (in memory, once loaded) | **METERS** always |
| `Results` | **METERS** |

`bemkit.geometry.load.SCALE_BY_SUFFIX` is the source of truth: `0.001` for
`.msh`, `1.0` otherwise.

## NC.inp Format

Emitted by `Deck.write` (`numcalckit/deck.py`); there is no template file.
Physical group IDs: `RIGID=1` (zero velocity, default), `VIBRATING=2`
(specified velocity). STEP colours: Blue (0,0,255) = rigid, Red (255,0,0) =
vibrating (read in `bemkit/geometry/generate.py`).

### Driven-group resolution

A foreign mesh names its physical groups its own way (Akabak writes
`HornWalls`/`SourceDisc`); `Group(name)` (`bemkit.model.select`) is the single
lookup, matching case-insensitively by *name*, never by tag id — foreign
meshes routinely invert bemkit's own `RIGID=1`/`VIBRATING=2` numbering.
Resolution happens once, at `model.resolved_bcs()` (called by every solver and
by `Model.vibrating_area`), and raises with the mesh's actual group names on a
miss — a surface with nothing driving it is a valid NumCalc job that returns
silence, so a wrong name must not resolve to "no boundary condition" quietly.

### NumCalc Boundary Condition Types

Each element carries one line: `ELEM e1 TO e2 <bctype> real(v0) curve1
imag(v0) curve2`.

| Type | Effect | v0=0 meaning |
|------|--------|--------------|
| `VELO` | Fixed particle velocity | Sound-hard (rigid) — default if unassigned |
| `PRES` | Fixed sound pressure | Sound-soft |
| `ADMI` | Acoustic admittance (1/impedance), ρ₀·c-normalised | Sound-absorbing |
| `IMPE` | Acoustic impedance, ρ₀·c-normalised (converted to admittance internally) | — |

- Elements with no assigned condition are implicitly sound-hard (`VELO=0`).
- An element may carry both `VELO` and `ADMI` — a vibrating source on an
  absorbing surface (one mesh element, two `BC`s with the same selector).
- `ADMI` supports frequency-dependent values via piecewise-linear `CURVES`;
  `curve1`/`curve2` are `-1` when not frequency-dependent. NumCalc requires
  all curve refs on a line to be `-1` or all non-`-1`.

### CURVES

Written by `model_to_deck` whenever an `Admittance`/`Impedance` condition is
frequency-dependent (an array or a callable — see The model layer, above): two
curves per condition, one for the real part and one for the imaginary, each
scaling its part of the constant `1.0` written on the `ELEM` line (a `0.0`
constant would silently zero that part instead). The curve abscissa is Hz,
interpolated linearly and clamped flat outside the given range — fully
decoupled from the frequency grid, so a curve is unaffected by a partial-band
solve.

Curves exist for `ADMI` only, and that is the physics, not an arbitrary
restriction: admittance/impedance enter the boundary *operator*, so the system
matrix genuinely changes per frequency, whereas `VELO`/`PRES` and every
incident source are the right-hand side of a linear system, whose
frequency-dependent response is a post-hoc scale on a unit-amplitude solve
(`Results.scaled_by` — see Results, above). NumCalc's own behaviour agrees:
it ignores a curve on `VELO`/`PRES` and `exit(-1)`s on `IMPE` with a curve,
telling you to use `ADMI` instead.

### PLANE WAVES / POINT SOURCES

`PlaneWave`/`PointSource` (`bemkit.model.sources`) pass through to `NC.inp` as
8-value rows: `[nr, x, y, z, ampl_real, -1, ampl_imag, -1]`. The middle three
fields mean different things per section — `PlaneWave.direction` is a travel
direction (need not be a unit vector; NumCalc normalises it, but it must not
be all zeros), `PointSource.position` is a metres location — which is exactly
why they are two Python types rather than one row shape. Curve refs are always
`-1`: frequency-dependent source amplitudes are not supported (post-scale the
`Results` instead, as above). Keep a point source ≥1 average edge length off
the surface (Hint 3) — `bem STUDY check` verifies this against every
`PointSource` in the model.

## Analytic Solution

`bemkit.model.analytic.sphcap_pressure` (also `bemkit.sphcap_pressure`, lazy —
needs the `analytic` extra) computes exact pressure for axisymmetric spherical
cap vibration via spherical-harmonic expansion:
`sphcap_pressure(eval_points, frequencies, c0, rho0, angle_deg, radius_mm,
nmax=None)` → raw pressure array. **`nmax` has a ceiling, not just a floor**:
`y_n(x) ~ -(2n-1)!!/x^(n+1)` overflows for `n` well past `x`, and `1j * inf` is
`nan + inf j`, so asking for a higher order past that point does not buy
accuracy — it returns NaN, and used to do so silently. `nmax=None` (the
default) picks `ceil(k_max · r0) + 20`, measured converged to 1e-12 against a
hand-picked 80 over 250 Hz–16 kHz; an explicit `nmax` that would overflow now
raises and names one that works. It is the reference `models/sphcap` and the
`BemppSolver` cross-validation test are both checked against.

## Visualization: `bemkit.view`

Inspection only (the `view` extra: pyvista + matplotlib, lazy-imported).
`BEMScene` renders 3D scenes, `ResultsViewer` drives the interactive `bem
STUDY results` viewer on top of it, `convergence.plot_convergence` draws the
solver-health figure (see `bem STUDY convergence`, above),
`bemkit.view.web.export_web` writes an auralizer bundle for the separate
`bemkit-auralizer` framework — but **`export-web` currently refuses**: it was
written against the config.toml stack and has not been ported to the `model.py`
spine, so it names the piece that is missing rather than failing somewhere
inside. The 2D plotting family for *results* (`bem MODEL plot`:
freq/ir/polar/isobaric/…) was removed on 2026-08-12 — recover it from git
history (`git show 2535eba:packages/bemviz/bemviz/viz/freq.py`) if it is wanted
back. `convergence` is not its return: it plots the solver's own logs, not
pressure, and takes plain `BinRecord` lists rather than a `Results`.

`bemkit.view` used to be the separate `bemviz` distribution; it is now a layer
inside `bemkit`, registered directly by `bemkit.cli.main` (not through the
`bemkit.cli` entry-point group, which is now only what `bemlab` uses).

## Reference

NumCalc source: `mesh2hrtf/NumCalc/src/NC_Input.cpp` in
https://github.com/Any2HRTF/Mesh2HRTF
