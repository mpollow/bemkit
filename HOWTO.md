# How to run a BEM simulation: model.py → results

A complete, worked walkthrough using the `bem` CLI (`bemkit`), from a
`model.py` you write to queryable acoustic results. The running example is a
vibrating spherical cap (`models/sphcap`), but the workflow is identical for
scanned heads, CAD assemblies, or any triangulated surface.

A study is Python:

```
model.py  →  Model  →  bem check  →  bem solve  →  Results
```

There is no config grammar. `model.py` calls plain Python functions from
`bemkit.model` to build a `Model`, and `bem` drives it through a solver.

---

## 0. Prerequisites (one-time)

```bash
# Install both packages + optional deps
pip install -e packages/numcalckit -e "packages/bemkit[all]"
#   ...or a subset: bemkit[meshing] for gmsh, bemkit[view] for the 3D viewers,
#   bemkit[analytic] for scipy (the sphcap analytic reference)
#   Without installing: use `python -m bemkit.cli ...` instead of `bem ...`

# NumCalc must be on PATH (the actual solver — not part of bemkit)
NumCalc -h        # if "command not found", add its bin/ dir to PATH
```

`bem <STUDY_DIR>` with no subcommand always prints a status report — run it
any time to see what stage each variant is at. `STUDY_DIR` defaults to `.`, so
running `bem <subcommand>` from inside a study directory works too (`bem
solve` ≡ `bem . solve`); a study directory that happens to share a name with a
subcommand (e.g. `check/`) needs the explicit `./` prefix to disambiguate.

---

## 1. Create the study directory

Everything for one study lives in `models/<name>/`. The only required file is
`model.py`:

```
models/sphcap/
├── model.py             # ← the only thing you must write by hand
├── mesh/                 # trusted meshes, however they got there (*.msh, *.stl, ...)
├── geometry/              # optional: CAD/scan sources, if model.py generates its own mesh
├── params.toml            # optional: flat scalars build() reads, not validated
└── runs/<solver>/<variant>/   # generated: NumCalc bundles + results, one tree per solver
```

`model.py` must define one of:

```python
def build() -> Model: ...
def build(params: dict) -> Model: ...    # params.toml, if present, passed as a dict
def variants() -> dict[str, Model]: ...  # wins over build() when both are defined
```

`bem` loads the module by path (not `sys.modules` — two studies named the same
thing never collide), calls `variants()` when present, and otherwise
`{"default": build()}` or `{"default": build(params)}` depending on `build`'s
signature. Whatever `build()`/`variants()` itself raises propagates with its
own traceback — a study is Python, and a `KeyError` for a missing params key
is a line number its author needs, not a message this layer could improve on.

---

## 2. Write `model.py`

The minimal shape — a rigid sphere with a vibrating cap:

```python
# models/sphcap/model.py
from pathlib import Path

from bemkit.geometry import load_mesh
from bemkit.model import BC, Group, Medium, Model, Velocity, circle, fft_grid, on_axis

HERE = Path(__file__).parent


def build(params):
    mesh = load_mesh(HERE / "mesh" / f"{params['mesh']}.msh")
    return Model(
        mesh,
        fft_grid(params["sr"], params["nbins"]),
        Medium(c=params["c"], rho=params["rho"]),
        bcs=[BC(Group("vibrating"), Velocity(1.0))],
        fields={
            "onaxis": on_axis("z", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
            "polar": circle("zx", res=5.0, radius=1.0),
        },
    )


def variants():
    """One model per mesh resolution — a loop, not a [variants] table."""
    base = dict(sr=32000, nbins=65, c=340.0, rho=1.2)
    return {name: build({**base, "mesh": name})
            for name in ("coarse", "standard", "fine")}
```

```toml
# params.toml — a flat dict, read with tomllib, not validated
sr = 32000
nbins = 65
c = 340.0
rho = 1.2
mesh = "standard"
```

Note that the study above does *not* ship one: `variants()` wins over `build()`,
and it supplies its own `base` dict, so nothing ever reads a `params.toml` next
to it. The file earns its place when `build()` is the entry point and you want
those scalars editable without touching Python.

`Model(mesh, frequencies, medium=None, *, bcs=(), sources=(), fields=None,
exterior=True, time_factor=-1.0)` is the whole contract — every argument below
is just a Python object you construct and pass in.

### Frequencies

`fft_grid(sr, nbins)` gives the uniform grid an impulse response needs
(`n_frequencies = nbins - 1`, DC excluded) — pass any other **strictly
ascending** Hz array for a frequency-response-only band:

```python
import numpy as np
frequencies = np.geomspace(100.0, 16000.0, 30)   # 30-point log sweep
```

A 30-point log sweep costs 30 solves instead of 64 and covers the bottom
octaves properly. Partial-band selection (`bem solve --freq/--bins/--preview`,
below) works the same either way — it indexes the array `Model.frequencies`
actually holds. **No time domain on a non-uniform band**: anything that needs
one (an inverse FFT of the whole spectrum) raises `IncompleteBand` — keep
`fft_grid` if you need impulse responses (`Model.is_uniform_band` tells you
which you have).

> **Units matter.** `Mesh` is always metres once loaded — `load_mesh` does the
> conversion (`.msh` is millimetres, GMSH's convention; `.stl`/`.ply`/`.obj`
> are already metres). Field points, selector anchors and source positions are
> all metres too.

### Getting a mesh

Three ways to end up with a `Mesh`:

**(A) You already trust a mesh file.** Just load it:

```python
mesh = load_mesh(HERE / "mesh" / "standard.msh")
```

**(B) Generate from CAD (`.step`).** GMSH reads STEP face colours — **blue
(0,0,255) → rigid, red (255,0,0) → vibrating** — and grades the mesh by
distance from the vibrating faces. There is no CLI verb for this: run it once,
by hand or from a small script, and commit the resulting `.msh`:

```python
from bemkit.geometry.generate import mesh_from_step
from bemkit.geometry.sizing import distance_field

mesh_from_step(
    HERE / "geometry" / "driver.step",
    HERE / "mesh" / "driver.msh",
    distance_field(lc_min=5.0, lc_max=25.0, dist_min=50.0, dist_max=100.0),
)
```

`lc_min`/`lc_max` are the finest/coarsest element edge (mm); the ramp between
them runs over `dist_min`–`dist_max` (mm) from the vibrating surface. Aim for
~6 elements per wavelength at the highest frequency you plan to solve
(`bem check`, below, tells you if you fell short).

**(C) Generate from a scan (`.ply`/`.stl`/`.obj`).** Same idea, no colour
information — the whole surface is graded as one group:

```python
from bemkit.geometry.generate import mesh_from_surface

mesh_from_surface(HERE / "geometry" / "head_scan.ply",
                   HERE / "mesh" / "head_scan.msh",
                   distance_field(5.0, 25.0, 50.0, 100.0))
```

**Geodesic sizing** — `bemkit.geometry.sizing.geodesic_field(lc_min, lc_max,
dist_min, dist_max)` — measures distance *along* the surface instead of
straight through space. Matters wherever the geometry folds back on itself:
on a horn loudspeaker, a point on the *outer* wall sits a few mm from the
piston through the air, but the surface path runs all the way around the horn
mouth, so `distance_field` refines the whole outer shell for no accuracy gain.
STEP/CAD sources only (a scan has no vibrating group to measure from — this
raises rather than silently falling back to Euclidean); needs scipy; costs an
extra coarse meshing pass to carry the distance computation. A non-watertight
model leaves some nodes unreachable — they are sized at `lc_max` with a loud
warning; a large fraction means the STEP is actually broken, not that the mesh
is coarse.

**Multiple resolutions or geometries** are just a Python loop —
`variants()` above is the pattern; there is no `[variants]` table to learn.

---

## 3. Drive the surface — boundary conditions

A `BC` pairs a **selector** (*where*) with a **condition** (*what*):

```python
from bemkit.model import BC, Group, Near, Where, Velocity, Pressure, Admittance, Impedance

BC(Group("vibrating"),              Velocity(1.0))              # a mesh physical group
BC(Near((0, 0, 0.05), radius=0.02), Velocity(1.0 + 0.5j))        # complex, geometric anchor
BC(Group("wall"),                   Admittance(felt))            # array over frequencies
BC(Where(lambda c: c[:, 2] < 0),    Pressure(0))                  # predicate on centroids
```

**Selectors:**

- `Group(name)` — a mesh physical group, matched **case-insensitively** by
  name. Foreign meshers rarely call the driven surface `"vibrating"` — Akabak
  writes `SourceDisc` — so just name it: `Group("SourceDisc")`. Raises,
  listing what the mesh does have, on a miss.
- `Near(point, radius=None)` — nearest element to a point (metres), or every
  element within `radius`. Anchors survive a remesh (they resolve at solve
  time, not export time), and are the friendliest form for a driven patch that
  should track the geometry across mesh resolutions. Raises if the anchor
  lands implausibly far from the surface, or a `radius` selects nothing
  (almost always a units/frame mistake).
- `Elements(indices)` — literal 0-based indices; use only when you actually
  mean one specific triangulation, since it goes stale on a remesh.
- `Where(predicate)` — `(M, 3) centroids -> (M,) bool`; for anything the other
  three cannot express.

**Conditions**, and what `v0=0` means for each:

| Condition | NumCalc keyword | `0` means | Frequency-dependent? |
|---|---|---|---|
| `Velocity(v=1.0)` | `VELO` | sound-hard (also the default for any unselected element) | no — see below |
| `Pressure(p=0.0)` | `PRES` | sound-soft | no |
| `Admittance(y=1.0)` | `ADMI` | — | **yes** |
| `Impedance(z=1.0)` | `IMPE` | — | **yes** |

`Admittance`/`Impedance` are **ρ₀·c-normalised** — `1.0` is a matched,
anechoic termination, not `1 m/(Pa·s)`. A magnitude above 100 is rejected
outright as almost certainly SI (divide an SI admittance by ~1/408, or an SI
impedance by ~408). An element may carry both `Admittance` and `Velocity` — a
vibrating source on an absorbing surface — by giving both `BC`s the same
selector. `TRAD`/`TRIM` (transfer admittance/impedance) are not offered:
NumCalc parses them and then hard-errors "not supported".

### Why only `Admittance`/`Impedance` take a spectrum

The BEM is linear, so a frequency-dependent **excitation** — `Velocity`,
`Pressure`, or an incident source's amplitude — is a scale factor on the
answer:

```python
res = solver.solve(model)              # unit-amplitude velocity
speaker = res.scaled_by(driver_response)   # (nf,) complex — no re-solve
```

Passing an array to `Velocity`/`Pressure` raises, naming `Results.scaled_by`.
`Admittance`/`Impedance` are different in kind — they sit in the boundary
*operator*, so the system matrix genuinely changes per frequency — and take
either an `(nf,)` array or a callable:

```python
felt = lambda f_hz: 0.05 + 0.0002j * f_hz          # or a plain array over model.frequencies
BC(Group("wall"), Admittance(felt))
```

Under the hood this becomes NumCalc `CURVES`: interpolated linearly in Hz and
clamped flat outside the given range, fully decoupled from the frequency
grid (so it is unaffected by a `--preview`/`--freq`/`--bins` partial solve).

### Incident sources (scattering)

Instead of, or in addition to, a vibrating surface:

```python
from bemkit.model import PlaneWave, PointSource

Model(mesh, frequencies, sources=[
    PlaneWave((0, 1, 0), amplitude=1.0),        # travelling towards +y; need not be unit length
    PointSource((0, -1, 0), amplitude=1.0),     # metres position
])
```

A plane wave has no location — it is defined by the *direction* it travels
in; a point source has no direction — it is defined by its *position*. Same
three numbers, different meaning, which is why they are two Python types
rather than one row shape you could transpose by mistake. Keep a point source
at least one average element edge length away from the surface
(arXiv:2312.00005, Hint 3): the free-field Green's function is singular at
zero distance, and `bem check` (below) verifies this. Source amplitudes are
frequency-independent for the same reason `Velocity`/`Pressure` are —
post-scale the `Results` instead of re-solving. Results default to NumCalc's
raw output, the natural reading for scattering; the acceleration
normalisation (`acceleration_scale`, needs a velocity drive) does not apply.

---

## 4. Define where to measure — field points

`Model.fields` is a plain `{name: (N, 3) array}` dict, metres:

```python
from bemkit.model import on_axis, circle, plane
import numpy as np

fields = {
    "onaxis":  on_axis("z", [0.1, 0.25, 0.5, 1.0, 2.0]),        # points along one axis
    "polar":   circle("zx", res=5.0, radius=1.0),                # circle in a plane, degrees step
    "planeXZ": plane("xz", res=0.05, limits=(-0.5, 0.5, -0.5, 0.5)),  # grid, metres
    "custom":  np.load(HERE / "fieldpoints" / "scan.npy"),        # anything you already have
}
```

No schema, no file discovery, no `exec` hook — composing, slicing, or writing
your own generator is ordinary numpy. Points on the wrong side of the surface
(inside it, for an exterior problem) or too close to it are dropped
automatically at solve time (ray-cast validation, unchanged from before).

---

## 5. Check the mesh before you spend hours on it

```bash
bem models/sphcap check
```

NumCalc validates almost nothing: it will happily solve a mesh with inverted
normals, a hole, or a band its elements cannot resolve, and hand back
plausible-looking numbers. `check` runs the tests it doesn't, over the `Model`
you built — there is no file to be missing here, since `build()` either
produced a `Model` or raised:

```
standard  (863 nodes, 1722 elements, up to 16000 Hz)
  ✓ watertight: closed surface, 1722 triangles
  ✓ orientation: consistent and outward (enclosed volume 0.0005161 m³)
  ✓ element_regularity: aspect ratio median 1.06, 99th pct 1.44, max 1.64
  ✓ size_uniformity: neighbour area ratio median 1.08, max 2.33
  ✗ elements_per_wavelength: 0.9 elements per wavelength at 16000 Hz ...
```

| Check | Catches |
|---|---|
| `watertight` | holes and non-manifold edges — the BEM problem is ill-posed without a closed surface |
| `orientation` | inconsistent winding, **and** normals pointing inward (Hint 1) |
| `elements_per_wavelength` | a mesh too coarse for the top of its own band (Hint 2) |
| `element_regularity` | slivers (Hint 4) |
| `size_uniformity` | abrupt size jumps between neighbours — what a graded mesh risks (Hint 6) |
| `source_clearance` | a point source sitting on the surface (Hint 3) |

`--strict` exits non-zero on warnings too, for CI. Errors always exit
non-zero. `--variant NAME` restricts to variants whose name starts with it.

> **Orientation is the one to take seriously.** NumCalc's own `-check_normals`
> only verifies that normals are *consistent* with each other — a uniformly
> inverted mesh passes it. `bem check` adds the signed-volume test for
> *outward*, which is the error the paper calls most common and which never
> announces itself in a plot.

Sanity-check visually too, before running (needs the `view` extra):

```bash
bem models/sphcap geometry          # mesh + BC colour + eval grid, in 3D
```

---

## 6. Solve

```bash
bem models/sphcap solve                  # all variants, one NumCalc at a time, in the background
bem models/sphcap solve -j 4             # at most 4 concurrent NumCalc processes
bem models/sphcap solve --force          # recompute all bins from scratch
bem models/sphcap solve --foreground     # wait for it instead of detaching
bem models/sphcap solve --restart        # kill a running NumCalc first
```

There is no separate `export` step: `solve` writes `runs/numcalc/<variant>/`
(a self-contained, relocatable bundle — NumCalc runs with it as CWD, writes
`be.out/` into it, and frequency bins are independent so runs parallelise and
resume trivially) as its own first move.

`solve` **starts in the background and returns.** NumCalc runs for hours, so
the terminal is not held for it: the command detaches a supervisor in its own
session — immune to the terminal closing and to Ctrl-C — and prints its pid,
the job count, and the log path. bemkit's own progress goes to
`runs/run.log`; NumCalc's per-variant `NC*.out` logs are unchanged. Nothing
extra is written to track the run, because nothing needs to be: `bem status`
finds live jobs with a `/proc` scan and reads completion out of `be.out/`
itself.

Anything decidable up front — NumCalc missing from PATH, a run already in
flight, no bins left to compute — is reported on the terminal before the
detach, so you never have to open the log to find out the run never started.
Use `--foreground` to wait instead.

With runs detached, `bem STUDY` status only tells you about the one study you
name. To see which project is computing across everything at once:

```bash
bem ps
```

```
NumCalc running (2 study/studies, 4 process(es)):
  hornstudio          horn_iwata_fc640_akabak_bem   22/64 bins   1 proc  pid 22881
  hornstudio          horn_os_fc400_akabak_bem      19/64 bins   1 proc  pid 22882
  hornstudio-quarter  hires_jmlcell_QUARTER_xy       4/64 bins   2 proc  pid 26817, 26818
```

`bem ps` ignores `STUDY_DIR` — it scans every NumCalc process on the machine
and reads study, variant and progress out of each one's working directory.

`-j N` is the only load knob: N is the number of NumCalc processes allowed to
run at once, whatever the study contains — **default 1**, because BEM peak
memory, not CPU, is the binding constraint: several concurrent DMLFMBEM solves
on a large mesh exhaust RAM. With fewer variants than slots, the planner fans
variants out over frequency ranges (`-istart`/`-iend`) until the slots are
full — the busiest variant gets the most, and because cost grows with
frequency the top-octave slices come out narrower than the bottom ones. Job
order alternates big and small so that a heavyweight and a lightweight run
side by side rather than two memory hogs at once.

`solve` computes only the bins missing from `be.out/`/`be.npz`, so a killed
run continues where it stopped — no bookkeeping file needed — and running
against complete results is a no-op. Use `--force` to recompute everything,
e.g. after `model.py` changes the mesh or the band.

**A resume also refuses if `model.py` changed underneath it.** Every existing
bin directory carries a fingerprint (`runs/.../model.sha256`) of the mesh,
band, medium, resolved boundary conditions and sources, and field-point sets
that produced it; a resume checks the current model against it and stops
rather than mixing two models' results in one `be.out/`. `--force` bypasses
this the same way it bypasses everything else — by discarding the bins first.

> **Convergence ≠ completion.** A bin can finish but not converge. `bem
> solve` and `bem status` scan `NC*.out` and surface warnings ("Maximum
> number of iterations is reached", "CGS failed"). These usually mean a
> defective mesh — **re-mesh** rather than raising iteration limits.

Check progress any time:

```bash
bem models/sphcap                   # status: per-variant bins done / missing / warnings,
                                    #   plus live NumCalc jobs (⟳ running, current bin)
```

`solve` **stops** if NumCalc is already working on one of the selected
variants — two processes in one directory would race each other's `be.out/`
writes and corrupt bins:

```
NumCalc already running for: fine (pid 26276, 26277)
  Wait for it, or re-run with --restart to kill and start over.
```

`--force` is not a way around that check — it deletes `be.out/` before
recomputing, so pairing it with a live solver is the one combination that has
to stay impossible. `--restart` is the explicit escape hatch: it terminates
the processes (SIGTERM, then SIGKILL) and then runs; on its own it *resumes*
(a bin killed mid-write fails the completeness check and is recomputed
anyway), so add `--force` only if you actually want everything redone.

You can also run NumCalc entirely outside bemkit:
`cd runs/numcalc/<variant> && NumCalc` (outputs `NC.out`, `be.out/`).

### Fast look first: partial-band runs

A full solve computes every bin. To see whether a model behaves sensibly,
solve a handful instead:

```bash
bem models/sphcap solve --preview 8       # 8 bins spread over the band
bem models/sphcap solve --freq 1000 4000  # just these frequencies (snapped to the grid)
bem models/sphcap solve --bins 4 16 64    # literal 1-based bin indices
```

The bins land on the model's own frequency grid, so nothing is wasted: a
later `bem models/sphcap solve` computes only what is still missing and the
results merge into one set. `bem status` shows a partial run as `◐  (8/64
bins)`. Frequency-domain results are usable straight away; anything crossing
the whole band (an impulse response) needs every bin and raises
`IncompleteBand` until you fill it in.

`--force` cannot be combined with a band selector — it discards every bin, so
the bins you did not name would still come from the previous model. To
restart a partial run from scratch, delete `runs/numcalc/<variant>/be.out/`
first.

### Will it fit in RAM?

```bash
bem models/sphcap estimate-ram        # clustering only, no solve — seconds
```

Reports NumCalc's own estimate at the **highest** frequency, where FMM memory
peaks. It under-estimates slightly (FMM matrices only, not the mesh and
result arrays), but it is the number to compare against free RAM before
committing to `-j 4`. `--all-bins` sweeps the whole band instead.

### Extra solver flags

```bash
bem models/sphcap solve --nitermax 5000     # max CGS iterations (NumCalc default 1000)
bem models/sphcap solve --check-normals     # verify normal-vector consistency
bem models/sphcap solve --adapt-fmmlength   # adapt multipole truncation to cluster radii
```

`--check-normals` verifies that every element normal points into the **same**
domain. It does *not* check they point outward, so a uniformly inverted mesh
passes — that check is `bem check`'s `orientation` (Hint 1), and it is still
on you before the first solve.

### Tuning the solver knobs — not yet wired up

NumCalc's own solver knobs — `bem_type` (TBEM/SLFMBEM/DMLFMBEM),
`isolver` (iterative vs. direct), `cluster_edge_length` (FMM bounding box),
`preconditioner` (incomplete-LU depth), `min_expansion_terms` — live on
`numcalckit.deck.Deck` with NumCalc's own defaults, and `Deck.validate` still
rejects the two combinations NumCalc accepts and silently downgrades
(`isolver=4` with an FMM `bem_type`; `preconditioner` alongside `isolver=4`).
But nothing on the `Model` → `NumCalcSolver` path sets them yet — there is no
successor to the old `[nc]` escape hatch. Until one exists, changing a knob
means calling the translation yourself instead of `NumCalcSolver.solve`:

```python
from bemkit.solvers.numcalc import model_to_deck
from bemkit.solvers import fieldpoints
from numcalckit.meshfiles import (
    write_object_mesh, write_eval_grid_nodes, write_eval_grid_elements,
)

raw_points, _ = fieldpoints.combine(model.fields)
mask = fieldpoints.validate(raw_points, model.mesh, exterior=model.exterior)
points, offsets = fieldpoints.combine(model.fields, mask)

write_object_mesh(model.mesh.vertices, model.mesh.triangles, workdir)
write_eval_grid_nodes(points, workdir)
n_eval_elements = write_eval_grid_elements(len(points), workdir)

deck = model_to_deck(model, n_eval_nodes=len(points), n_eval_elements=n_eval_elements)
deck.cluster_edge_length = 0.05      # any Deck field
deck.write(workdir / "NC.inp")
# then run NumCalc yourself, e.g. `numcalckit.run_jobs`, or `cd workdir && NumCalc`
```

Reach for this only when a run actually misbehaves:

| Symptom | Knob |
|---|---|
| A large mesh exhausts RAM, or `NC.out` reports expansion order `L > 30` | `cluster_edge_length` — shrink the FMM bounding box (try `0.05`) |
| The iterative solver converges slowly or not at all | `preconditioner` — negative values select incomplete-LU depth (`-1`, `-2`, `-3`) |
| Same, or you want a reference answer on a small mesh | `isolver = 4` with `bem_type = 0` — the direct Gauss solver, O(N³) |
| Accuracy at high frequency | `min_expansion_terms` — 5..10, default 8 |

Non-convergence usually means a defective mesh — re-mesh before reaching for
these.

---

## 7. Pack results (optional but recommended)

```bash
bem models/sphcap pack                  # be.out/ → be.npz (~4× smaller), be.out/ removed
bem models/sphcap pack --keep-original  # pack, but leave be.out/ in place
```

`be.npz` shadows `be.out/` everywhere downstream. (A subsequent full `solve`
removes a stale `be.npz` so it can't hide fresh output — re-`pack`
afterwards.)

**`pack` removes `be.out/` by default.** It is >99% of a variant directory
and pure redundancy once packed, and the removal is not taken on trust: every
bin and quantity is re-read from the ASCII tree and compared against the
packed arrays first, and any mismatch keeps `be.out/` and raises. A variant
with a live NumCalc is skipped rather than packed — the other variants of the
same study still pack.

---

## 8. Look at the results

> **2D plots were removed on 2026-08-12.** `bem MODEL plot` and the
> freq/ir/polar/isobaric/plane/... family are gone; they were not useful in
> practice. Use the `Results` API (below) to get the arrays and plot whatever
> you actually want, or recover the old module from git history.

### Interactive 3D field/surface inspection (needs the `view` extra)

```bash
bem models/sphcap results --freq 1000                 # surface + all field-point sets
bem models/sphcap results onaxis --freq 1000          # restrict to one set
bem models/sphcap results onaxis --freq 1000 --variant fine
#   --quantity magnitude_db|magnitude|phase|real|imag
#   --norm raw|acc   (acc needs a velocity-driven model; refused otherwise, not silently shown as raw)
#   --intensity  (acoustic intensity arrows)   --impedance   --plane   --screenshot out.png
```

`results`/`geometry` export any variant whose bundle is missing before
drawing it — there is no separate export step to run first.

### Export a web auralizer bundle

```bash
bem models/sphcap export-web --out bundle/
```

**Currently refuses**: `export_web` was written against the config.toml stack
and has not been ported to the `model.py` spine yet — it names the missing
piece rather than failing partway through. The signal-processing half of the
module (resampling, EQ derivation, quantisation) is intact and still tested;
only the model.py-driven entry point is missing.

### Programmatic access — `Results`

```python
from bemkit.study import load_study, workdir
from bemkit.results import Results, acceleration_scale

models = load_study("models/sphcap")
wd = workdir("models/sphcap", "standard")     # runs/numcalc/standard/
res = Results.load(wd / "results.npz")        # what a finished `bem solve` leaves behind

res.frequencies           # (nf,) Hz
res.set_names             # ['onaxis', 'polar']
res.pressure("onaxis")    # (N, nf) complex — raw pressure (NumCalc's own output)
res.velocity("onaxis")    # (N, nf, 3) complex
res.boundary_pressure()   # (M, nf) complex — surface pressure
res.boundary_velocity()   # (M, nf) complex — surface normal velocity

acc = res.scaled_by(acceleration_scale(models["standard"]))  # ÷ (j·rho0·ω·A_vib)
acc.pressure("onaxis")    # same shape, acc-normalized — raises NoVelocityDrive if
                          #   the model has no velocity drive (a scattering model)
```

`res.materialise()` applies any pending scale eagerly and turns every further
read into a plain dict lookup. `res.save(path)` / `Results.load(path)`
round-trip through `.npz` (`path` may be a directory, in which case
`results.npz` is used). In practice, `bem solve` already leaves
`runs/numcalc/<variant>/results.npz` behind — `Results.load` on that path is
usually all you need.

---

## The whole thing, end to end

```bash
# 1. write models/sphcap/model.py (mesh + BCs/sources + field points),
#    and params.toml if build() is the entry point and takes params

bem models/sphcap check                        # 2. mesh diagnostics before spending hours
bem models/sphcap geometry                      # 3. eyeball mesh + BCs + eval grid (3D)
bem models/sphcap solve -j 4                     # 4. solve (detaches; -j bounds concurrent NumCalc)
bem models/sphcap pack                           # 5. shrink be.out/ -> be.npz

bem models/sphcap                                # 6. status: everything complete, no warnings?
bem models/sphcap results onaxis --freq 1000     # 7. inspect the field in 3D
```
