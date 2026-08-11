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
├── fieldpoints/          # optional pre-made eval points (*.npy, METERS)
├── *.py|.sh              # optional scripts, referenced explicitly via exec = ... in config.toml
├── numcalc/<variant>/    # generated: NumCalc job bundles + results
│                         #   (snapshot.json records the config that produced them)
└── plots/                # generated: figures
```

### `config.toml` — minimal but complete

```toml
[bem]
sr = 32000            # sample rate (Hz) → sets the frequency grid
nbins = 65            # n_frequencies = nbins - 1  (DC bin excluded)
                      # freqs = linspace(0, sr/2, nbins)[1:]

air_density   = 1.2   # rho0
speed_of_sound = 340.0
bem_type = 4          # 4 = DMLFMBEM, O(N log N)  [recommended default]
time_factor = -1.0    # e^{-iωt} convention
is_internal = 0       # 0 = exterior (radiation) problem

# Solver knobs — every default below is NumCalc's own, so omitting them all
# changes nothing. Reach for them when a run misbehaves, not before. See
# "Tuning the solver" further down.
cluster_edge_length = 0.0   # ClusEdgL0 (m); 0 = NumCalc default. Shrink when a
                            #   large mesh exhausts RAM.
preconditioner = 0          # IPRECOND in [-3, 2]; negative = incomplete-LU depth
min_expansion_terms = 0     # 0 = NumCalc default (8); otherwise 5..10
solver = 0                  # ISOLVER: 0 iterative (CGS), 4 direct (needs bem_type = 0)

# Mesh sizing (mm). Aim for ~6 elements per wavelength at the highest frequency.
[mesh]
lc_min = 5.0          # finest element edge
lc_max = 25.0         # coarsest element edge (far from the source)
dist_min = 50.0       # size ramps from lc_min→lc_max between these distances
dist_max = 100.0
distance = "euclidean"   # or "geodesic" — see below (STEP/CAD sources only)

# Where to evaluate the field: declarative generator sets (see step 3).
[fieldpoints.onaxis]
type = "onaxis"
dim = "z"
radii = [0.1, 0.25, 0.5, 1.0, 2.0]   # meters
```

> **Units matter.** Fieldpoint `.npy` files are in **meters**. `.msh` meshes are
> **millimeters** (GMSH convention); `.stl`/`.ply`/`.mesh` are **meters**. The
> pipeline converts automatically based on file extension.

---

## 2. Provide geometry — pick ONE of three entry points

`bem MODEL mesh` first runs the `[geometry]` exec script if one is configured
(e.g. a `bemgeo` script that writes STEP files), then meshes every source it
finds in `geometry/` into a surface `.msh` in `mesh/`. The split is strict:
`geometry/` holds things that still need meshing; anything in `mesh/` is
trusted and used as-is, never remeshed. A source is skipped when its mesh
already exists.

```toml
[geometry]
exec = "./geometry_box_speaker.py"   # runs with cwd = model dir; writes geometry/*.step
```

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

**Starting from nothing but a mesh?** `bem init` writes the `config.toml` for
you, measured against the mesh itself:

```bash
bem mymodel init path/to/model.msh   # copies into mesh/, writes config.toml
bem mymodel export
bem mymodel run
```

It picks fieldpoint radii that clear the mesh bounding box (fixed radii would
land *inside* a large or off-centre mesh and be dropped by validation), and it
works out what drives the model:

- **A `vibrating` physical group** (what bemkit's own mesher writes) — used
  directly, nothing to configure.
- **Groups with other names** — the common case for a mesh from another tool.
  bemkit matches the group by name, so `SourceDisc` or `Piston` is invisible to
  it by default. `init` picks the *smallest* group (a source patch is a small
  fraction of the radiating body) and writes it down explicitly:

  ```toml
  [groups]
  vibrating = "SourceDisc"   # every element outside this is rigid
  ```

  Check that line — if the guess is wrong, edit it rather than re-running init.
- **No usable groups at all** — a **placeholder** `[BOUNDARY]` anchor on the
  topmost element, flagged loudly. **Edit it before running**; see step 4.

The reason for the noise: a surface with nothing driving it is a perfectly
valid NumCalc job that returns silence. Export therefore *raises* if a
configured `[groups] vibrating` name is not in the mesh, and warns if the mesh
has groups but none of them resolved.

> **Variants = mesh files.** Every mesh file under `mesh/` becomes a variant with
> its own `numcalc/<stem>/` bundle. One file → one variant; many files → many
> variants meshed/exported/run/packed independently. (You can also define
> BC-only variants that share a mesh via a `[variants]` table — see CLAUDE.md.)

### Geodesic sizing — `distance = "geodesic"`

By default the size ramp measures straight-line distance from the vibrating
surface. That is wrong wherever the geometry folds back on itself. On a horn
loudspeaker, a point on the *outer* wall sits a few mm from the piston through
the air, but the surface path runs all the way around the horn mouth — so the
whole outer shell gets refined as if it were radiating, inflating element count
and solve time for no accuracy gain.

```toml
[mesh]
lc_min = 5.0
lc_max = 25.0
dist_min = 50.0
dist_max = 100.0
distance = "geodesic"   # measure over the surface, not through the air
```

Two worked examples ship with the repo, both built as two variants that differ
only in this key:

| example | geometry | effect |
|---|---|---|
| `examples/horn` | free-standing horn | 6600 → 4472 elements (−32%) |
| `examples/hornbox` | horn recessed in a cabinet | 17180 → 8370 elements (−51%), with BEM results for both |

`examples/hornbox` is the one to read: it shows *why* the metric matters rather
than just that it does. Resolving a 280 mm flare needs `dist_max ≈ 350`, and a
euclidean 350 mm reaches straight through the cabinet wall to the back face
70 mm behind the diaphragm — so you cannot ask for a well-resolved horn without
also refining the entire box.

The four size knobs keep their exact meaning; only the metric changes. Notes:

- **STEP/CAD sources only.** Scan surfaces (`.ply`/`.stl`/`.obj`) carry no
  vibrating region to measure from, so `bem mesh` raises rather than silently
  meshing as euclidean.
- **Costs an extra meshing pass.** A coarse pilot mesh is generated first to
  carry the distance computation, then discarded. On a model where the
  euclidean and geodesic distances agree (anything convex — a sphere, a flat
  baffle) it buys nothing; the gain is entirely in folded, occluded, or
  re-entrant geometry.
- **Needs a watertight model.** Distances travel over the surface, so a region
  on a disconnected shell can't be reached at all. Those regions are sized at
  `lc_max` and `bem mesh` prints how many nodes were affected — a large
  percentage means the STEP is broken, not that the mesh is coarse.
- Requires scipy (`pip install -e ".[meshing]"`).

### Multi-variant meshing — per-variant `[mesh]` overrides
To run the same geometry at several mesh resolutions, name the variants in a
`[variants]` table; each entry may override `[mesh]` parameters. The mesh is
generated as `mesh/<variant>.msh` from the `geometry/` source (the sole source
by default; pick one with `source = "<stem>"` when there are several):

```toml
[mesh]              # shared defaults
lc_min = 5.0
lc_max = 25.0

[variants.fine]
mesh = { lc_min = 2.5, lc_max = 12.0 }

[variants.coarse]
mesh = { lc_min = 10.0, lc_max = 50.0 }
```

Geometry sweeps work the other way around: have the `[geometry]` exec script
write several `.step` files — every source becomes its own variant.

---

## 3. Define where to measure — fieldpoints

Fieldpoints are the coordinates (in **meters**) where you want the radiated
pressure. The common shapes are declared directly in `config.toml` — each
`[fieldpoints.<name>]` table is one named set, and the keys after `type` are
the generator's arguments (`make_fields_onaxis` / `polar` / `plane`):

```toml
[fieldpoints.onaxis]
type = "onaxis"
dim = "z"
radii = [0.1, 0.25, 0.5, 1.0, 2.0]   # meters

[fieldpoints.polar]
type = "polar"
dims = "zx"
res = 5                              # degrees

[fieldpoints.planeXZ]
type = "plane"
dims = "xz"
limits = [-500, 500, -500, 500]      # mm
res = 10                             # mm
```

Pre-made point clouds work too — either name them
(`[fieldpoints.custom]` with `file = "fieldpoints/custom.npy"`), or define no
sets at all and every `fieldpoints/*.npy` is loaded as one set per file. For
anything custom, an escape hatch script can generate the files at export time:

```toml
[fieldpoints]
exec = "./make_fieldpoints.py"   # runs before export; writes fieldpoints/*.npy
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

A top-level `velocity = 0.5` in `config.toml` sets the default velocity for
geometric entries that don't carry their own `velocity`/`v` (it defaults
to 1.0).

`[BOUNDARY]`, `[SYMMETRY]`, `["PLANE WAVES"]` and `["POINT SOURCES"]` are
validated when the config loads, exactly like `[bem]`/`[mesh]`: a malformed
entry (`cap = [100, 100]` missing its velocity) or a typo'd key
(`radios = 0.01`) fails immediately, naming the offending entry — it can
never again silently export a rigid surface that solves to silence.

### Non-rigid surfaces — `type` and `value`

By default every element is sound-hard and a `[BOUNDARY]` entry drives it with a
velocity. A geometric entry can carry any of NumCalc's four condition types
instead:

```toml
[BOUNDARY]
piston   = { near = [0, 0, 0.05], velocity = 1.0 }              # VELO (shorthand)
driver   = { near = [0, 0, 0.05], type = "velo", value = [1.0, 0.5] }  # complex
soft     = { near = [0, 0.1, 0], type = "pres" }                # sound-soft (0 Pa)
absorber = { near = [0, 0, -0.1], radius = 0.05,
             type = "admi", value = [0.4, -0.2] }               # absorbing
wall     = { near = [0.2, 0, 0], type = "impe", value = [12.0, -3.0] }
```

| `type` | Meaning | Default `value` |
|---|---|---|
| `velo` | normal velocity (the default) | `1.0` |
| `pres` | sound pressure — `0` is sound-soft | `0.0` |
| `admi` | **normalized** admittance | `1.0` |
| `impe` | **normalized** impedance | `1.0` |

`value` is one number or `[real, imag]`. Give `velocity`/`v` *or* `type`/`value`,
not both. Element selection is unchanged — anchors and radii work exactly as they
do for a velocity condition, so an absorbing patch survives a remesh too.

**`admi` and `impe` are normalized by ρ₀·c**, not SI. `admi = 1.0` is a matched,
anechoic termination; an SI admittance is wrong by a factor of ~410 and looks
entirely plausible on a plot, so values above 100 are rejected outright. Divide an
SI admittance by ~1/408, or an SI impedance by ~408.

An element may carry both `admi` and `velo` — a vibrating source on an absorbing
surface. Write two entries with the same anchor.

`TRAD`/`TRIM` (transfer admittance/impedance) are not offered: NumCalc parses them
and then exits with "not supported".

#### Frequency-dependent absorption — `[curves]`

A real material is not one number. Point an `admi` entry at a named curve:

```toml
[BOUNDARY]
absorber = { near = [0, 0, -0.1], radius = 0.05, type = "admi", curve = "felt" }

[curves.felt]
frequencies = [125, 250, 500, 1000, 2000, 4000]   # Hz, ascending
real        = [0.05, 0.12, 0.30, 0.52, 0.61, 0.66]
imag        = [-0.02, -0.06, -0.14, -0.21, -0.24, -0.25]   # optional
```

Values are interpolated linearly in Hz and held flat outside the range, so the
curve need not match `sr`/`nbins` — it is decoupled from the frequency grid
entirely. `imag` defaults to zero.

**Curves are `admi`-only.** NumCalc ignores them for `velo`/`pres`, and `impe`
plus a curve is a hard `exit(-1)` that tells you to use `admi` — so bemkit
refuses all three at config load.

### Incident sources (scattering) — plane waves / point sources
Instead of (or in addition to) a vibrating surface, incident fields pass
straight through to `NC.inp` as 8-value rows:

**The middle three values mean different things in the two sections.** A plane
wave has no location — it is defined by the *direction* it travels in. A point
source has no direction — it is defined by its *position*. The row layout is
identical, so a row copied from one section to the other is accepted and wrong.

```toml
["PLANE WAVES"]
#     nr, dx, dy, dz, ampl_real, -1, ampl_imag, -1
1 = [  1,  0,  1,  0,     1,     -1,     0,     -1]   # travelling towards +y

["POINT SOURCES"]
#     nr, x0, y0, z0, ampl_real, -1, ampl_imag, -1
1 = [  1,  0, -1,  0,     1,     -1,     0,     -1]   # source at y = -1 m
```

The plane-wave direction need not be a unit vector — NumCalc normalizes it —
but it must not be all zeros. Point-source positions are in meters, like
fieldpoints.

Keep a point source at least one average element edge length away from the
surface (arXiv:2312.00005, Hint 3): the free-field Green's function has a
singularity at zero distance, so a source on or near the surface causes the
element-subdivision loop to fail or return nonsense. To drive the surface
itself, use a velocity boundary condition instead.

Amplitudes are frequency-independent (`CURVES` are not supported; the curve-ref
slots are always written as `-1`). Results default to NumCalc's raw output,
which is the natural reading for scattering setups; the acceleration-BC
normalization (`norm="acc"`, which assumes a velocity BC) is opt-in.

Sanity-check the setup in 3D **before** running:

```bash
bem examples/sphcap geometry          # mesh + driven (red) elements + eval grid
```

### Mirror planes — `[SYMMETRY]`

NumCalc can reflect the boundary elements about coordinate-aligned planes.
That buys you two different things depending on what your mesh represents:

- **Symmetry** — the object *and* its excitation are mirror-symmetric, so you
  mesh only the symmetric fraction and the images complete the object. Half
  the surface means a quarter of the system matrix.
- **A reflecting plane** — an infinite rigid or pressure-release plane (a
  baffle, a floor). Here you mesh the *whole* object beside the plane and the
  images are fictitious: this is the method of images building a half-space.
  The object need not be symmetric about the plane at all.

```toml
[SYMMETRY]
planes = [1, 1, 0]              # per axis, one plane each — see table below
coordinates = [0.0, 0.0, 0.0]   # metres — where each plane cuts its axis
```

| code | meaning | you mesh | the images are | field about the plane |
|------|---------|----------|----------------|-----------------------|
| 0 | no plane on this axis | — | — | — |
| 1 | symmetric plane | half the object | its other half | even (sound-hard) |
| 2 | antisymmetric plane | half the object | its other half | odd (sound-soft) |
| 3 | hard reflecting plane | the whole object | fictitious | even (sound-hard) |
| 4 | soft reflecting plane | the whole object | fictitious | odd (sound-soft) |

`num_symmetric_planes` in `NC.inp` is derived from the nonzero entries; you
never set it yourself.

**The magnitude says what the images mean, not whether they exist.** The
boundary elements are mirrored in all four cases, identically: NumCalc builds
`2**k` reflected clusters from the same element list regardless of the code
(`NC_Addresses.cpp:558`). What the magnitude records is whether the reflected
copy is *real surface* — the other half of a symmetric object (1/2) — or a
*virtual image* standing in for an infinite plane (3/4). In the second case the
object you meshed is the entire physical object; nothing of it lies beyond the
plane.

**So codes 3/4 compute exactly what 1/2 do.** Only the sign reaches the solver:
every read of the plane code tests `== 0`, `!= 0` or `< 0`, never the magnitude
(`NC_3dFunctions.cpp:239`, `NC_Addresses.cpp:581`). That is correct rather than
sloppy — mirror symmetry and the method of images are the same operation, so
the solver can forget which one you meant once it has the sign. Pick the code
that documents your intent. (Code 5, an impedance ground plane, is described in
`NC_Arrays.h` but rejected by the parser — it does not exist.)

**At most three planes, one per coordinate axis.** `planes` is indexed by axis,
so `[1, 1, 0]` is an x-plane and a y-plane and there is no way to express a
second plane normal to the same axis. Two parallel planes would anyway need an
infinite image series; NumCalc builds exactly `2**k` copies for `k` planes
(`numReflectionsOfElements_ = pow(2, numSymmetricPlanes_)`, capped at 8), which
cannot represent a periodic array. For a line array, use planes *perpendicular*
to each other — one through the array axis, one across its midpoint — for the
usual 4x reduction.

The mesh must sit entirely on one side of every active plane and must have
**no elements lying in a plane** — a cut face that got capped would be
reflected onto itself.

Evaluation points may go anywhere, including into the mirrored region:
NumCalc reflects the *elements*, so the field is evaluated correctly
throughout space. `export` validates them against the mirrored geometry, so
a point that lands inside or on a mirrored surface is dropped exactly like
one that hits the meshed surface. Without that, NumCalc dies mid-run with
`nsbe must be < MSBE` while subdividing the element under the point.

With a **reflecting** plane, though, only one side is physical. The solver
evaluates the mirror-symmetric field everywhere and validation will accept a
point behind the baffle, because it is outside the geometry — but the value
there describes the fictitious image, not your model. Keep evaluation points on
the object's side of a reflecting plane.

**Both 3D viewers draw the images.** `bem MODEL geometry` and
`bem MODEL results` render the reflected copies as a second surface, so a
symmetry-reduced model shows the whole object rather than a lone quadrant
sitting next to a full-space eval grid. The ghosts carry mirrored data, not a
neutral shell: boundary-condition colours in `geometry`, surface pressure in
`results` — sign-flipped once per antisymmetric plane crossed, which is a pi
shift you can see in `--quantity phase` but not in a magnitude. The symmetry is
read from the variant's `NC.inp`, so the picture shows what was actually
exported. Pass `--no-mirror` to either command to see only the elements you
meshed. A model without `[SYMMETRY]` is unaffected.

**What marks the simulated elements is the wireframe, not transparency.** Both
viewers draw black mesh edges on the elements NumCalc actually solved and leave
the ghosts smooth — so a full model is covered in edges, a quarter model carries
them on one quadrant. The lines are wide but nearly transparent (`line_width` 3
at 10 % opacity), which shades each element boundary rather than drawing a hard
outline on it: the mesh density stays readable at a glance without the
wireframe competing with the colour it sits on, and it does not turn into a
solid black mat where a fine mesh is viewed at a distance. `--no-edges` turns
them off in either viewer.

Ghosts are *not* faded, and both viewers default to an opaque surface, because
a see-through surface superimposes the far side onto the near side. That is
harmless for a bare mesh but actively misleading for surface pressure: on a
horn the throat and the outside of the mouth differ by tens of dB, and blending
them produces a picture of neither. `--mesh-opacity` still lowers it if you
want to see eval points inside the object; just do not read pressure off a
transparent surface.

---

## 4b. Check the mesh before you spend hours on it

```bash
bem examples/sphcap check
```

NumCalc validates almost nothing: it will happily solve a mesh with inverted
normals, a hole, or a band its elements cannot resolve, and hand back
plausible-looking numbers. `check` runs the tests it doesn't:

```
standard  (standard.msh: 863 nodes, 1722 elements)
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

`--strict` exits non-zero on warnings too, for CI. Errors always exit non-zero.

> **Orientation is the one to take seriously.** NumCalc's own `-check_normals`
> only verifies that normals are *consistent* with each other — a uniformly
> inverted mesh passes it. `bem check` adds the signed-volume test for
> *outward*, which is the error the paper calls most common and which never
> announces itself in a plot.

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
bem examples/sphcap run                       # all variants, one NumCalc at a time, in the background
bem examples/sphcap run -j 4                  # at most 4 concurrent NumCalc procs
bem examples/sphcap run --force               # recompute all bins from scratch
bem examples/sphcap run --foreground          # wait for it instead of detaching
bem examples/sphcap run --restart             # kill a running NumCalc first
```

`run` **starts in the background and returns.** NumCalc runs for hours, so the
terminal is not held for it: the command detaches a supervisor in its own
session — immune to the terminal closing and to Ctrl-C — and prints its pid, the
job count and the log path. bemkit's own progress goes to `numcalc/run.log`;
NumCalc's per-variant `NC*.out` logs are unchanged. Nothing is written to track
the run, because nothing needs to be: `bem status` finds live jobs with a `/proc`
scan and reads completion out of `be.out/` itself.

Anything decidable up front — NumCalc missing from PATH, a run already in
flight, no bins left to compute — is reported on the terminal before the detach,
so you never have to open the log to find out the run never started. Use
`--foreground` to wait instead (`bem all` always does).

With runs detached, `bem MODEL` status only tells you about the one model you
name. To see which project is computing across everything at once:

```bash
bem ps
```

```
NumCalc running (2 model(s), 4 process(es)):
  hornstudio          horn_iwata_fc640_akabak_bem   22/64 bins   1 proc  pid 22881
  hornstudio          horn_os_fc400_akabak_bem      19/64 bins   1 proc  pid 22882
  hornstudio-quarter  hires_jmlcell_QUARTER_xy       4/64 bins   2 proc  pid 26817, 26818
```

`bem ps` ignores `MODEL_DIR` — it scans every NumCalc process on the machine and
reads model, variant and progress out of each one's working directory, so it
works the same from any directory.

`-j N` is the only load knob: N is the number of NumCalc processes allowed to
run at once, whatever the model contains — **default 1**, because BEM peak
memory, not CPU, is the binding constraint: several concurrent DMLFMBEM solves
on a large mesh exhaust RAM. With fewer variants than slots, the planner fans
variants out over frequency ranges (`-istart`/`-iend`) until the slots are
full — the busiest variant gets the most, and because cost grows with
frequency the top-octave slices come out narrower than the bottom ones. Job
order alternates big and small so that a heavyweight and a lightweight run side
by side rather than two memory hogs at once.

`run` computes only the bins missing from `be.out/`/`be.npz`, so a killed run
continues where it stopped — no bookkeeping file needed — and running against
complete results is a no-op. The fan-out follows suit: ranges are cut from the
*missing* bins and never span one that already has results. Use `--force` to
recompute everything, e.g. after re-exporting a changed mesh or config.

> **Convergence ≠ completion.** A bin can finish but not converge. `bem run` and
> `bem status` scan `NC*.out` and surface warnings
> ("Maximum number of iterations is reached", "CGS failed"). These usually mean a
> defective mesh — **re-mesh** rather than raising iteration limits.

Check progress any time:

```bash
bem examples/sphcap                   # status: per-variant bins done / missing / warnings,
                                      #   plus live NumCalc jobs (⟳ running, current bin)
```

`bem run` **stops** if NumCalc is already working on one of the selected
variants — two processes in one directory would race each other's `be.out/`
writes and corrupt bins. It names the pids and does nothing else:

```
NumCalc already running for: fine (pid 26276, 26277)
  Wait for it, or re-run with --restart to kill and start over.
```

`--force` is not a way around that check. It deletes `be.out/` before
recomputing, so pairing it with a live solver is the one combination that has to
stay impossible. `--restart` is the explicit escape hatch: it terminates the
processes (SIGTERM, then SIGKILL) and then runs. On its own it *resumes* — a bin
killed mid-write fails the completeness check and is recomputed anyway — so add
`--force` only if you actually want everything redone.

You can also run NumCalc entirely outside bemkit:
`cd numcalc/<variant> && NumCalc` (outputs `NC.out`, `be.out/`).

### Fast look first: partial-band runs

A full run solves every bin. To see whether a model behaves sensibly, solve a
handful instead:

```bash
bem examples/sphcap run --preview 8      # 8 bins spread over the band
bem examples/sphcap run --freq 1000 4000 # just these frequencies
```

The bins land on the model's normal frequency grid, so nothing is wasted: a
later `bem examples/sphcap run` computes only what is still missing and the
results merge into one set.

`bem status` shows a partial run as `◐  (8/64 bins)`. Frequency-domain plots
work on partial results and mark the computed bins; time-domain plots
(`plot ir`, `plot etc`) need the whole band and are skipped with a message
until you fill it in.

`--force` cannot be combined with a selection — it discards every bin, so the
bins you did not name would still come from the previous export. To restart a
partial run from scratch, delete `numcalc/<variant>/be.out/` first.

### Will it fit in RAM?

```bash
bem examples/sphcap estimate-ram      # clustering only, no solve — seconds
```

Reports NumCalc's own estimate at the **highest** frequency, where FMM memory
peaks. It under-estimates slightly (FMM matrices only, not the mesh and result
arrays), but it is the number to compare against free RAM before committing to
`-j 4`. `--all-bins` sweeps the whole band instead.

If the answer is too big, shrink `cluster_edge_length` (below) before reaching
for a coarser mesh.

### Extra solver flags

```bash
bem MODEL run --nitermax 5000     # max CGS iterations (NumCalc default 1000)
bem MODEL run --check-normals     # verify normal-vector consistency
bem MODEL run --adapt-fmmlength   # adapt multipole truncation to cluster radii
```

`--check-normals` verifies that every element normal points into the **same**
domain. It does *not* check they point outward, so a uniformly inverted mesh
passes — that check is still on you (Hint 1).

### Tuning the solver

The four `[bem]` solver knobs all default to NumCalc's own values, so leaving
them out is the same as before they existed. Change one when a run misbehaves:

| Symptom | Knob |
|---|---|
| A large mesh exhausts RAM, or `bem status` reports expansion order `L > 30` | `cluster_edge_length` — shrink the FMM bounding box (try `0.05`). The documented remedy in arXiv:2312.00005 §3.3.2. |
| The iterative solver converges slowly or not at all | `preconditioner` — negative values select incomplete-LU depth (`-1`, `-2`, `-3`) |
| Same, or you want a reference answer on a small mesh | `solver = 4` with `bem_type = 0` — the direct Gauss solver, O(N³) |
| Accuracy at high frequency | `min_expansion_terms` — 5..10, default 8 |

Two combinations are rejected at config load, because NumCalc accepts them and
then quietly does something else: `solver = 4` with any FMM `bem_type` (it
falls back to iterative), and `preconditioner` alongside `solver = 4` (it is
forced to 2). Non-convergence usually means a defective mesh — re-mesh before
reaching for these.

**Anything not listed above** can still be set, via the raw `[nc]` table, which
names fields on the NumCalc input deck directly:

```toml
[nc]
min_expansion_terms = 9
isolver = 0
```

`[nc]` is applied last and wins over `[bem]`. It only accepts scalar deck
fields: values that are *derived* (frequencies, boundary conditions, sources,
symmetry, mesh counts) must be set in the section they come from, so
`config.toml` keeps describing what was actually exported. From Python, the
same escape hatch is just an attribute:

```python
from bemkit.sim.builder import deck_from_config
deck = deck_from_config(config, "myvariant", counts)
deck.cluster_edge_length = 0.05
deck.write(nc_dir / "NC.inp")
```

---

## 7. Pack results (optional but recommended)

```bash
bem examples/sphcap pack                  # be.out/ → be.npz (~4× smaller), be.out/ removed
bem examples/sphcap pack --keep-original  # pack, but leave be.out/ in place
```

`be.npz` shadows `be.out/` everywhere downstream. (A subsequent full `run`
removes a stale `be.npz` so it can't hide fresh output — re-`pack` afterwards.)

**`pack` removes `be.out/` by default.** It is >99 % of a variant directory and
pure redundancy once packed, and the removal is not taken on trust: every bin and
quantity is re-read from the ASCII tree and compared against the packed arrays
first, and any mismatch keeps `be.out/` and raises. `complex64` is lossless
relative to NumCalc's 6-digit text output, and `unpack_variant` regenerates the
ASCII tree from `be.npz` if you ever need it back. Pass `--keep-original` to skip
the removal. A variant with a live NumCalc is skipped rather than packed — the
other variants of the same model still pack.

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
bem examples/sphcap results --freq 1000                        # surface + all fieldpoint sets
bem examples/sphcap results onaxis --freq 1000                 # restrict to one set
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

sim.p_onaxis             # (N, nf) complex — raw pressure (NumCalc output, the default)
sim.p_acc_onaxis         # (N, nf) complex — normalized ÷ (j·rho0·ω·A_vib), opt-in
sim.nodes_onaxis         # (N, 3) meters
sim.boundary_pressure()  # (M, nf) surface pressure (raw; norm="acc" opt-in)
result = sim.result("onaxis")   # adapter for the viz functions
```

The base data is always what NumCalc computed (raw, velocity-BC output).
The `acc` normalization — which assumes a velocity BC and a vibrating
area — is requested explicitly via `norm="acc"` or the `p_acc_<set>` /
`v_acc_<set>` attributes. `SimulationData` is the public results API;
the `BEMResult` class it hands to viz functions is internal.

---

## The whole thing, end to end

```bash
# 1. author examples/sphcap/config.toml (fieldpoint sets + BC or STEP colors),
#    drop geometry in geometry/ or mesh/ — or point [geometry].exec at a script

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
