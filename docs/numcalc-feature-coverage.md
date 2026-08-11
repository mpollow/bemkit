# NumCalc feature coverage — what bemkit exposes, what it doesn't

**Date:** 2026-08-11 (supersedes the 2026-07-26 revision)
**Sources:** Kreuzer, Pollack, Brinkmann & Majdak, *NumCalc: An open-source BEM
code for solving acoustic scattering problems*, arXiv:2312.00005v3 (2024-10-31),
cross-checked against the NumCalc solver at `mesh2hrtf/NumCalc/src/` —
`NC_Input.cpp`, `NC_Main.cpp`, `NC_EquationSystem.cpp` — and against Mesh2HRTF's
own Python exporter, `mesh2hrtf/Mesh2Input/`.

Where the paper and the source disagree, **the source wins** and the
disagreement is noted. The same rule now applies to this document's own history:
the 2026-07-26 revision was written against `NC_Input.cpp` alone, and reading the
other two translation units contradicted it twice. Those corrections are marked
**[corrected]** below.

## Summary

Incident sources are not a missing feature — `bem` already writes plane waves and
point sources into `NC.inp` correctly. Boundary-condition *types* are the visible
gap, but they are not the largest one.

The largest gap is that **bemkit cannot express a non-uniform frequency list.**
NumCalc's frequency grid is a piecewise-linear curve evaluated in Hz (§1.6);
bemkit hardwires a two-point curve that collapses to the uniform FFT grid. That
is the right default for HRTF and impulse-response work and should stay the
default, but log-spaced or third-octave sweeps are currently inexpressible, and
nothing in the config surface hints that they are possible.

The second finding is architectural and is new to this revision (§4). Full
NumCalc coverage is *small* — roughly 300–500 lines of writer and validation
against bemkit's current 10 631. What did not scale was the **threading cost**:
`write_nc_inp` took a `ModelConfig`, so every new solver knob meant edits in
`config.py`, a resolver, the writer, the `NC.inp` template, the config snapshot,
the tests and `HOWTO.md`. That is now fixed — `bemkit/numcalc/` is a deck module
with no inward imports (§4, built in §5.1), so a new setting is one field plus
one line each in `write()`/`read()`, and the layer stays extractable as a
standalone `pynumcalc` if it ever earns its own distribution.

---

## 1. What NumCalc actually accepts

### 1.1 Boundary conditions

Line grammar: `NC_Input.cpp:809-877`. Keyword dispatch: `NC_Input.cpp:1227-1259`.
(The `NC_ReadBoundaryConditions` at `:883` is a deprecated copy that scans for
`IMAG`/`NEGA` keywords; it is commented out and does not run. Read only the
version at `:792`.)

```
ELEM e1 TO e2 <bctype> real(v0) curve1 imag(v0) curve2
```

| Keyword | `nbtyp` | Meaning |
|---|---|---|
| `VELO` | 0 | Particle velocity normal to the element. `0` ⇒ sound-hard (the default for any element with no BC line). |
| `PRES` | 1 | Sound pressure. `0` ⇒ sound-soft. |
| `ADMI` | 2 | **Normalized** surface admittance — see §1.2. |
| `IMPE` | 2 | Normalized impedance. Converted internally to admittance: `Y = 1/Z`, i.e. `re/(re²+im²)`, `-im/(re²+im²)`. |
| `TRAD`, `TRIM` | 3 | **Parsed, then hard-error**: "Transfer admittance/impedance boundary condition is not supported!" Do not expose. |

Behaviours worth knowing, none of which are in the paper's prose:

- **`IMPE` with value `0+0i` silently becomes `PRES` sound-soft** (`:1269-1272`).
- **`IMPE` combined with a curve is a hard `exit(-1)`**, with the message
  "Please use ADMI conditions. They work with CURVES." So a frequency-dependent
  surface must be expressed as admittance, never impedance.
- **An element may carry both `ADMI` and `VELO`** — this is how a vibrating
  source on an absorbing surface is modelled (paper, Appendix A).
- A BC line needs **at least 9 whitespace-separated terms** (`:833`).

### 1.2 `ADMI` is ρc-normalized, not SI

The single highest-risk item in this document. NumCalc expects
**admittance ÷ (ρ₀·c)** — dimensionless — so `Y = 1.0` is a matched, anechoic
termination, not `1 m/(Pa·s)`.

Two independent confirmations:

- `NC_EquationSystem.cpp:1914-1915` applies `gammfact` to `zbval1` under the
  comment *"for surface (normalized) admittance boundary condition"*, where
  `gammfact = waveNumbers_ * harmonicTimeFactor_` (`:1835`).
- Mesh2HRTF's own `Mesh2Input/Materials/write_material.py:27-31`:
  *"The admittance must be normalized, i.e., admittance/(rho*c) has to be
  provided."*

`IMPE` inherits the same normalization, since NumCalc merely reciprocates it.

Passing an SI admittance yields a result that is wrong by a factor of ~410 and
looks entirely plausible on a plot. Whatever config surface exposes `ADMI` must
name the convention in the key itself or validate the magnitude.

### 1.3 Curve references are all-or-nothing (`NC_Input.cpp:844-849`)

```
ERROR: Curves have to be specified for all parts of a Boundary or Source.
I.e., ALL curve numbers in one line must be -1 or ALL curve numbers must
not be -1
```

This applies to **boundary conditions and sources alike**. A frequency-dependent
real part with a constant imaginary part is rejected. Any future `CURVES`
support must enforce the pairing at config-validation time rather than letting
NumCalc fail at solve time.

### 1.4 The `CURVES` section — abscissa is Hz **[corrected]**

The previous revision claimed curve nodes are *indexed by frequency step*, which
made a curve "only meaningful alongside the `sr`/`nbins` grid" and coupled to
partial-band runs. **That is wrong.** `NC_Input.cpp:1141` reads the abscissa as a
frequency, and `NC_EquationSystem.cpp:1997-2018` interpolates on `frequency_` in
Hz, clamped flat below the first node and above the last.

**Curves are completely decoupled from the frequency grid**, and therefore from
partial-band runs. This removes the only stated risk in the old Tier 3.

Format (reference implementation: `Mesh2Input/mesh2input.py:1346-1367`), written
immediately before `POST PROCESS`:

```
CURVES
<n_curves> <max_points_per_curve>
<curve_id> <n_points>
<freq_hz> <value> 0.0        # third field is read and ignored
...
```

The curve carries a real scale factor applied to that part of the BC value. A
complex frequency-dependent admittance therefore needs **two** curves, one per
part, scaling real and imaginary independently — an arbitrary complex `Y(f)`
cannot be expressed with fewer. Mesh2HRTF's exporter writes the constant as
`1.0` on the BC line and lets the curves carry the whole value.

### 1.5 Incident sources (`NC_Input.cpp:1002-1069`)

```
PLANE WAVES
Nr  DX DY DZ  Real(P0) curve1 Imag(P0) curve2

POINT SOURCES
Nr  X0 Y0 Z0  Real(P0) curve1 Imag(P0) curve2
```

- Both require **exactly 8 terms**; anything else is a hard error.
- **The parser only compares the first token** — `chterms[0].compare("PLANE")`
  (`:1017`) and `chterms[0].compare("POINT")` (`:1046`). So `PLANE` and
  `PLANE WAVES` are equally valid. The paper's Appendix C.4 documents
  `POINT`/`PLANE` while its own example listing uses `PLANE WAVES`; both work,
  and **bemkit's current spelling is correct**.
- **For a plane wave, fields 1–3 are a direction, not a position**, and NumCalc
  **normalises it itself** (`Vcnorm_dim3_(dirinw[i], dw)`, `:1034`). The user need not
  supply a unit vector — but a zero vector is a division by zero.
- For a point source, fields 1–3 are a position in metres.
- **`numPointSources_ = -1` is a reserved internal value**: it generates a
  built-in "test point source" used to check the accuracy of the program
  (`NC_Main.cpp:347`). Never expose it as a user-facing count.

### 1.6 The frequency grid is a curve, not a range — bemkit's largest gap

Three pieces combine to define which frequencies are solved:

1. **Controlparameter II** (`NC_Input.cpp:207-209`) supplies `numFrequencies_`,
   `frequencyStepIncrement_` and `firstFrequencyStep_`.
2. **Load Frequency Curve** supplies `npoicurv` pairs of `time frequency`.
3. **`NC_ComputeFrequencies`** (`NC_Input.cpp:1686`) maps step *i* to
   `t = first + (i+1)·increment`, then **piecewise-linearly interpolates the
   curve** at `t` to get the frequency in Hz — clamped flat outside its range.

bemkit used to write a fixed two-point curve with `increment = 1e-6` and
`first = 0`, spanning `(0, 0)` to `(n_freq·1e-6, sr/2)`, so the interpolation
collapsed to the uniform FFT grid

```
f_i = (i+1) · sr / (2 · (nbins − 1))
```

which is exactly right for HRTF and impulse-response work and remains the
default that `sim/builder.py` builds. Since Tier 0 the *format* no longer
constrains it: `Deck.frequencies` is an explicit Hz array and the curve is
written with one node per step (`numcalc/deck.py`), so **log-spaced,
third-octave or fully arbitrary frequency lists are already representable** —
what is still missing is a config surface to ask for one, and an `analysis/`
answer for what the band-crossing plots do with it.

This is the one gap a raw escape hatch cannot fully paper over, because
`analysis/` assumes an FFT grid: `plot ir` and `plot etc` inverse-transform the
band, and `config.freq_bin`/`resolve_bins` snap onto a uniform grid. Supporting
non-uniform lists means deciding what those functions do when handed one —
almost certainly raise the existing `IncompleteBand`-style refusal with a clear
reason, the same pattern partial-band runs already established.

### 1.7 Main Parameters I is richer than we use (`NC_Input.cpp:247-253`)

bemkit writes `2 {n_elem} {n_nodes} 0 {n_sym} 2 2 {bem_type} 0`. Four of the nine
fields were undocumented here:

| Field | Variable | Notes | bemkit |
|---|---|---|---|
| 0 | `numElementGroups_` | element-group count | hardcoded `2` |
| 1,2 | `numElements_`, `numNodes_` | must be ≥ 1 | computed |
| 3 | `minExpansionTermsFMM_` | **Minimum FMM expansion terms.** `0` ⇒ default 8; must otherwise be in `[5, 10]` or hard error. | `[bem] min_expansion_terms` |
| 4 | `numSymmetricPlanes_` | must be in `[0, 3]` | computed |
| 5,6 | — | read, never used | `2 2` |
| 7 | `methodBEM_` | `0` TBEM, `1` SLFMBEM, `4` DMLFMBEM | `[bem] bem_type` |
| 8 | `methodSolver_` | **ISOLVER**: `0` iterative (CGS) or `4` direct (Gauss); anything else is a hard error. **Silently forced to `0` whenever FMM is active** (`:290`). | `[bem] solver` — the silent fallback is rejected at config load |

`minExpansionTermsFMM_` is the knob behind Hint 8 (expansion order `L > 30`) —
the paper gives the hint, and this is one of the two levers that move it (the
other is `-adapt_fmmlength`, §1.10).

`methodSolver_ = 4` is only reachable with `bem_type = 0` (TBEM), which makes it
a small-mesh reference-solution tool rather than a production setting. Selecting
it also forces `methodPreconditioner_ = 2` (`:323`).

### 1.8 Main Parameters II (`NC_Input.cpp:295-323`)

```
nplanewaves  npointsources  isInternal  ClusEdgL0  IPRECOND
```

| Field | Variable | Notes |
|---|---|---|
| 3 | `ClusEdgL0_` | **FMM initial bounding-box edge length**, metres. `0.0` ⇒ NumCalc's default, `(√N·A₀)^½`. Exposed as `[bem] cluster_edge_length`. |
| 4 | `methodPreconditioner_` | Must be in `[-3, 2]`. Negative values select incomplete-LU scanning degree: `-1`→1, `-2`→2, `≤-3`→3, each with `methodPreconditioner_=0`. Forced to `2` when the direct solver is used. Exposed as `[bem] preconditioner`. |

Only these five fields are read; trailing values are ignored.

`ClusEdgL0_` is not a minor knob. The paper's Conclusions name shrinking the
initial bounding box as *the* remedy when the multipole expansion length `L`
blows up memory on large meshes (Sec. 3.3.1, Sec. 4.2, Conclusions), and
Sec. 4.2 says a large `L` also destabilises the solver.

### 1.9 Main Parameters III and IV

**Main Parameters III** (`NC_Input.cpp:325-326`) is consumed and **entirely
discarded** — no assignment of any kind. It is a required placeholder line and
nothing more. Documented here so nobody hunts for its meaning again.

**Main Parameters IV** (`NC_Input.cpp:329-336`) reads only three fields:
`speedOfSound_`, `densityOfMedium_`, `harmonicTimeFactor_`. Note that
**`harmonicTimeFactor_` is coerced to ±1** — `if (harmonicTimeFactor_ != -1.0)
harmonicTimeFactor_ = 1.0` (`:334`) — so it is a convention switch, not a scale.
Fields 3 and beyond are ignored, and the solver tolerance `epsilon_` is hardcoded
to `1e-6` (`:340`), not configurable from the deck.

Both a speed of sound below 300 and a density below 1.0 produce warnings, not
errors — so an accidental unit mistake in `[bem]` is survivable but silent in
`NC.inp` terms.

### 1.10 Command line (`NC_Main.cpp:150-186`)

The complete flag set:

| Flag | Effect | bemkit |
|---|---|---|
| `-istart N` / `-iend M` | 1-based inclusive bin range | ✅ `numcalc/runner.py` |
| `-nitermax N` | max CGS iterations | ❌ |
| `-estimate_ram` | clustering only, no solve; writes `Memory.txt` | ❌ |
| `-check_normals` | half-edge orientation consistency check | ❌ |
| `-adapt_fmmlength` | adapt multipole truncation to cluster radii | ❌ |
| `-h` | usage | — |

An **unrecognised flag makes NumCalc `exit(-1)`**, so a runner that passes
through unknown options must validate them itself.

`-adapt_fmmlength` is the second lever on Hint 8, alongside `ClusEdgL0`.

### 1.11 Prior art: the only other Python writer requires Blender

`mesh2hrtf/Mesh2Input/mesh2input.py` is the sole existing Python NC.inp writer:
1389 lines, `import bpy` at module scope, and the writer itself lives inside a
`bpy.types.Operator.execute` method. **It cannot be imported as a library.**

It remains the best available reference for the `CURVES` format (§1.4) and for
the material CSV convention (`Materials/write_material.py`), but there is no
reusable code to adopt. This matters for §4: a dependency-light NC.inp
reader/writer is a genuine hole in the ecosystem, not a duplicate.

### 1.12 What must never be exposed

- `TRAD` / `TRIM` — parsed, then hard-error.
- `numPointSources_ = -1` — internal accuracy self-test (§1.5).
- Main Parameters III — read and discarded (§1.9).

---

## 2. What bemkit does today

Post-Tier-0 (§5.1): the *format* side is complete for everything below — each ❌
is now a missing **config surface** or builder mapping, not a missing capability
in the writer. `Deck` already carries the fields; nothing asks for them yet.

| NumCalc capability | bemkit | Where |
|---|---|---|
| `VELO`, real value | ✅ | `sim/builder.py` → `Deck.boundary` |
| `VELO`, complex value | ✅ `value = [re, im]` | `sim/boundary.py` |
| `PRES` | ✅ `type = "pres"` | `sim/boundary.py` |
| `ADMI` / `IMPE` | ✅ `type = "admi"`/`"impe"`; SI-magnitude guard | `config.py:_check_bc_type` |
| `ADMI` + `VELO` on one element | ✅ two entries, same anchor | `sim/boundary.py` |
| `CURVES` (frequency-dependent) | ✅ `[curves]` + `curve =` on `admi` | `sim/builder.py` |
| Plane waves | ⚠️ works, raw 8-value pass-through | `sim/builder.py:_sources` |
| Point sources | ⚠️ same | same |
| Element selection by geometry | ✅ and better than NumCalc needs | `sim/boundary.py` |
| Non-uniform frequency list | ⚠️ deck holds explicit Hz; builder always builds the FFT grid | `Deck.frequencies` |
| `minExpansionTermsFMM_` | ✅ `[bem]` key + `[nc]` | `sim/builder.py` → `Deck.min_expansion_terms` |
| `ISOLVER` (`methodSolver_`) | ✅ `[bem]` key + `[nc]` | `sim/builder.py` → `Deck.isolver` |
| `ClusEdgL0` | ✅ `[bem]` key + `[nc]` | `sim/builder.py` → `Deck.cluster_edge_length` |
| Preconditioner | ✅ `[bem]` key + `[nc]` | `sim/builder.py` → `Deck.preconditioner` |
| `-nitermax` | ✅ `bem run --nitermax` | `numcalc/runner.py:solver_flags` |
| `-estimate_ram` | ✅ `bem MODEL estimate-ram` | `cli/_pipeline.py` |
| `-check_normals` | ✅ `bem run --check-normals` | `numcalc/runner.py:solver_flags` |
| `-adapt_fmmlength` | ✅ `bem run --adapt-fmmlength` | `numcalc/runner.py:solver_flags` |
| `-istart` / `-iend` | ✅ | `numcalc/runner.py` |
| Solver-convergence check (Hint 5) | ✅ | `numcalc/status.py:solver_warnings` |

**Both accuracy problems flagged in the 2026-07-26 revision are closed.**
Plane-wave fields are documented as a *direction* in `HOWTO.md:339-356` and
enforced by `_SOURCE_TRIPLE` (`sim/builder.py`); the `ClusEdgL0`/`IPRECOND`
comment is now emitted by `Deck.write` and is correct. No action needed.

---

## 3. The ten hints, mapped to bemkit

| # | Hint | bemkit status |
|---|---|---|
| 1 | Normal vectors point away from the scatterer | ⚠️ **[corrected]** — see below. NumCalc's own partial check is now reachable via `bem run --check-normals`; the *outward* question is still unanswered. |
| 2 | 6–8 elements per wavelength | ✅ `bem check` compares the longest edge against the top of the band. Previously implicit: `[mesh] lc_max` is documented as "~6 elements per wavelength at max freq", but nothing verifies it against `sr`/`nbins` after meshing. `element_areas` already exists. |
| 3 | ≥ 1 average edge length between an external source and the surface | ✅ `bem check` (point sources). Previously **not checked** — and this is the validation that matters most for incident sources. bemkit already computes exactly this quantity for fieldpoints (`min_surface_distance_factor × avg_edge_length`); point sources need the same test. |
| 4 | Keep elements as regular as possible | ✅ `bem check` reports a normalized aspect ratio (1.0 = equilateral). |
| 5 | Check the iterative solver converged | ✅ `solver_warnings` scans `NC*.out` for both strings; surfaced in `bem status` and after a run. |
| 6 | Elements approximately the same size, at least locally | ✅ `bem check` reports the neighbour area ratio — directly relevant to the graded meshes bemkit produces. |
| 7 | Subdivision-loop error ⇒ overlaps, or a source too close | ❌ the error string is not in `_SOLVER_WARNINGS`. One-line addition to an existing mechanism. |
| 8 | Expansion order `L > 30` ⇒ check convergence and plausibility | ⚠️ `L` is reported per level in `NC.out` and is still not parsed, so nothing tells you to act. Both levers are now reachable: `[bem] min_expansion_terms` and `[bem] cluster_edge_length` (§1.7, §1.8); `-adapt_fmmlength` (§1.10) is not. |
| 9 | Check the plausibility of your results | — user's job, but sparse/partial results now make this easier. |
| — | Mesh must be watertight, each edge in exactly two elements | ✅ `bem check` (holes and non-manifold edges). |

**Hint 1 correction.** The previous revision stated that "NumCalc does not check
either". It does — behind an opt-in flag. `-check_normals` (`NC_Main.cpp:181`)
routes to `NC_checkedges()` (`NC_Input.cpp:741`, called at `:516`), which builds a
half-edge tally: for every node it accumulates signed incoming and outgoing
half-edge indices and requires the sum to be zero, skipping evaluation elements.

Two limits matter, and neither is obvious from the flag's name:

- It verifies that all normals point into the **same** domain, **not** that they
  point outward. A uniformly inverted mesh passes cleanly.
- It is off by default, so nobody gets it for free.

So bemkit still has something to add here — a signed-volume test on a watertight
mesh answers the *outward* question that `-check_normals` does not — but the
honest framing is "NumCalc offers a partial check we do not invoke", not "NumCalc
offers nothing".

Hints 1, 3, 4, 6 and 7 are all *mesh or setup* diagnostics that bemkit is well
placed to provide and currently does not. They could ship as a `bem MODEL check`
family (the CLI reserves that name for the private `bemlab` package — worth
reconsidering).

---

## 4. Architecture: does a standalone `pynumcalc` make sense?

The question behind this document: if bemkit grows to cover every NumCalc
setting, on top of meshing, analysis, visualization, a CLI and a web exporter,
does that justify splitting the NumCalc layer into its own package?

**Decision: build an extract-ready subpackage inside bemkit now; do not split the
repository until a named trigger fires.**

### 4.1 The size fear is misplaced

Full grammar coverage is roughly **300–500 lines** of writer plus validation.
For scale, bemkit is 10 631 lines today, of which `viz/` alone is ~2 700 and
`cli/` ~1 630. Feature-completeness against NumCalc is not what would make bemkit
big.

### 4.2 The coupling was the real problem

*(Diagnosis as of 2026-08-11, before Tier 0; kept because it is the rationale for
the structure that replaced it.)*

`write_nc_inp(config, variant, nc_dir, ...)` took a `ModelConfig`. **The NC.inp
writer was config.toml-shaped**, so each new knob cost edits in `config.py`
(schema + validation), a resolver or the writer, the `NC.inp` template,
`sim/config_snapshot.py`, the tests, and `HOWTO.md`. Six touch points per knob,
across four layers, is what scales badly — not the knob count.

The extraction seam did **not exist**, contrary to what the layer rule implied.
`CLAUDE.md` documented only `sim ↛ analysis/viz`, and `sim/` did respect that —
but it had undeclared inward dependencies that the extraction had to cut first:

- `sim/discovery.py` and `sim/boundary.py` import `bemkit.config`
- the mesh writers, `sim/boundary.py`, `sim/export.py` and `sim/fieldpoints.py`
  import `bemkit.meshing.io`

Of these, only the mesh writers were on the NumCalc side of the line; the rest
are builder-layer code that was never meant to move. That is why the fix below
turned out small.

### 4.3 The fix is a document model, not a repository

`bemkit/numcalc/` (built — §5.1): a typed, faithful, round-trippable representation of
an NC.inp deck that knows nothing about config.toml, meshes, variants or bemkit.

```
bemkit/numcalc/          numpy-only, zero bemkit.* imports
  deck.py       Deck dataclass — frequencies (explicit Hz), medium, solver,
                symmetry, boundary, plane_waves, point_sources, curves
                + Deck.read(path) / Deck.write(path)
  grammar.py    per-section emit + validation; one place per NumCalc knob

bemkit/sim/builder.py    deck_from_config(config, variant, mesh) -> Deck
                         bemkit's opinions live HERE: the FFT frequency grid,
                         physical-group -> VELO, mm/m unit scaling
```

`write_nc_inp` becomes a thin shim over `builder` + `Deck.write`, keeping its
current signature. Adding a NumCalc feature then costs: one field, one line in
`write()`, one in `read()`, one round-trip test.

Four things this buys:

1. **Full coverage without config bloat.** `config.toml` keeps only curated keys
   with a real user story. A raw `[nc]` table and a Python hook
   (`deck = deck_from_config(...); deck.solver.nitermax = 5000`) reach the other
   100% without a TOML key per NumCalc field. This is what makes "support
   everything" compatible with "keep the config legible".
2. **The extraction seam, for free.** A module with no inward imports becomes
   `pynumcalc` via `git mv` plus a compatibility shim, on the day it is wanted.
3. **Immediate value, not deferred.** `Deck.read()` collapses today's two ad-hoc
   NC.inp re-parsers — `read_nc_inp_boundary_conditions` (`sim/readers.py:135`)
   and `read_nc_inp_symmetry` (`:195`) — into one tested reader. The variant
   directory is already a self-contained bundle; a real reader is what makes it
   introspectable.
4. **Round-trip equality is the test that makes completeness tractable.**
   `read(write(deck)) == deck` over a corpus of decks is a single property that
   covers every field at once. Without it, "all settings" is untestable and will
   rot.

**Enforcement.** `tests/test_boundaries.py` already carries
`CORE = ["config.py", "meshing", "sim"]` (`:18`) and walks the package for
forbidden imports. Extend it with a rule that nothing under `bemkit/numcalc/`
imports `bemkit.*`. Without that check the seam closes again within a month and
the extraction option is quietly lost.

### 4.4 Why not a separate repository now

- **No second consumer exists.** `bemlab` extends the *CLI* through the
  `bemkit.cli` entry-point group (`cli/__init__.py:52-54`), not the writer;
  `bemkit-auralizer` consumes web bundles, not decks. A package with one consumer
  is a module.
- **Repository count is already four** — bemkit, bemgeo, bemlab,
  bemkit-auralizer. A fifth must earn its place.
- **A premature split has real costs**: version skew, a second CI and changelog,
  and a cross-repo pull request for every feature touching both sides. Nearly
  every increment in §5 touches both sides.

### 4.5 The extraction trigger

Naming the trigger is what makes deferral honest rather than indefinite. Extract
`pynumcalc` as its own distribution when **either**:

- a second consumer needs decks without bemkit's meshing and analysis stack, **or**
- the deck library is offered upstream or published — for which there is a real
  ecosystem gap, since the only existing Python writer requires Blender (§1.11).

Until then the subpackage boundary delivers the same design benefit at none of
the coordination cost.

### 4.6 What stays in bemkit regardless

Two things are bemkit's own value and must not migrate into a generic deck
library:

- **Geometric BC anchors** (`sim/boundary.py`) — `near=`/`radius=` resolution
  against the mesh at export time. NumCalc has no concept of these; they are the
  reason a BC survives a remesh.
- **The mesh-quality diagnostics of §3.** NumCalc performs none of them by
  default and, as the paper puts it, "will produce *some* results, and it is up to
  the user to check the validity". That is the gap where bemkit adds value the
  solver does not — and it needs the meshing layer, so it could never live in
  `pynumcalc` anyway.

---

## 5. Roadmap

Re-sequenced around the deck refactor and the corrected sizing.

| | Work | Size | Note |
|---|---|---|---|
| 0 | `bemkit/numcalc/` deck + round-trip tests; `write_nc_inp` removed; boundary-test rule | medium | ✅ **Landed.** See §5.1. |
| 1 | Solver knobs — `ClusEdgL0`, `IPRECOND`, `min_expansion_terms`, `ISOLVER` — plus the `[nc]` escape hatch | small | Two of the four are newly identified (§1.7). |
| 2 | `PRES` / `ADMI` / `IMPE`, complex values, `ADMI`+`VELO` on one element | small | ✅ **Landed.** `type`/`value` on a geometric entry; values > 100 rejected as SI. |
| 3 | Normalization semantics for non-velocity-driven models | small + decision | ✅ **Landed.** See §5.2. |
| 4 | Typed incident sources + Hint 3 distance validation | small | Unchanged from the previous revision. |
| 5 | `CURVES` | small | ✅ **Landed.** `[curves.<name>]` + `curve = "<name>"` on an `admi` entry. |
| 6 | CLI flags — `-nitermax`, `-check_normals`, `-estimate_ram`, `-adapt_fmmlength` | small | ✅ **Landed.** `bem run --nitermax/--check-normals/--adapt-fmmlength`, `bem MODEL estimate-ram`. |
| 7 | Non-uniform frequency lists | medium | New. Needs an `analysis/` decision (§1.6). |
| 8 | Mesh-quality diagnostics (`bem MODEL check`) — hints 1, 2, 3, 4, 6 | medium | ✅ **Landed.** `bemkit/meshing/quality.py` + `bem MODEL check`. Hint 7 (the subdivision-loop string) is still unparsed. |

Tiers 1–6 are additive and touch one layer each once Tier 0 lands.

### 5.1 Tier 0, as built

`bemkit/numcalc/` now holds the NumCalc format and process layer — `deck.py`,
`meshfiles.py`, `readers.py`, `packing.py`, `runner.py`, `status.py` — and imports
nothing from bemkit, on stdlib and numpy only. Both rules are enforced by
`tests/test_boundaries.py`. `sim/builder.py` holds every bemkit opinion; `write_nc_inp`
and the `templates/NC.inp` package data are gone.

Three things worth recording for whoever picks up Tier 1:

- **Adding a setting is now one `Deck` field plus one line each in `write()`/`read()`.**
  `min_expansion_terms`, `isolver`, `cluster_edge_length` and `preconditioner` already
  exist as fields carrying their previously-hardcoded values, so Tier 1 is wiring a
  config key to a field, not surgery.
- **The frequency grid is no longer baked into the format.** `Deck.frequencies` is an
  explicit Hz array and the Load Frequency Curve is written with one node per step, so
  Tier 7 (non-uniform lists, §1.6) is a builder change plus an `analysis/` decision —
  the deck already supports it. The general encoding is also *exact*, where the old
  two-point shorthand reached each step through a division and asked for
  250.00000000000003 Hz when it meant 250.
- **`read_nc_inp_boundary_conditions` stayed a lenient section scanner** rather than
  becoming a `Deck.read` wrapper. It feeds `A_vib` and therefore every `norm="acc"`
  result, and its tests deliberately exercise NC.inp *fragments* that a strict deck
  parser rejects. It now shares `deck.parse_bc_line`, which was the part actually worth
  deduplicating.

### 5.3 Tier 1, as built

Four curated `[bem]` keys — `solver`, `min_expansion_terms`,
`cluster_edge_length`, `preconditioner` — plus a raw `[nc]` table that sets any
scalar `Deck` field directly and is applied last. Every default is NumCalc's
own, so a config that sets none of them produces byte-identical output to
before, which is what the golden test checks.

Three notes for later tiers:

- **The curated/raw split works as designed.** Adding these four cost one line
  each in `_BEM_KEYS`, the `ModelConfig` dataclass and the builder — the deck
  already had the fields from Tier 0. The `[nc]` hatch means a knob with no
  user story (say `n_element_groups`) needs no schema entry at all.
- **`[nc]` rejects *derived* fields by name**, pointing at the section they
  really come from (`frequencies` → `[bem] sr/nbins`, `boundary` →
  `[BOUNDARY]`). Letting a raw table shadow those would make `config.toml` stop
  describing what was exported, which is the one thing an escape hatch must not
  cost.
- **Two combinations are rejected that NumCalc merely ignores**: `solver = 4`
  with an FMM `bem_type` (silently downgraded, `NC_Input.cpp:290`) and
  `preconditioner` alongside `solver = 4` (forced to 2, `:323`). Both are
  checked at config load *and* in `Deck.validate`, so the `[nc]` hatch and
  direct Python callers cannot route around them.

Verified against the real solver: NumCalc's own log echoes
`Minimum number of terms in the FMM expansion. = 6` and the incomplete-LU
preconditioner for a deck built from these keys
(`tests/numcalc/test_solver_integration.py`).

### 5.2 The normalization decision (Tier 3), as built

**Decided: `acc` is refused, not faked, and `"auto"` picks the honest one.**

`normalize_pressure` used to *warn* and return the raw array when `A_vib` was
None. A scattering model therefore produced raw Pascals wearing an
acceleration-normalized label — wrong by a frequency-dependent factor, and
invisible on a plot. It now raises `NoVelocityDrive`, whose message names
`norm="raw"` as the quantity that exists.

To keep that from making scattering models unplottable, the whole stack learned
a third norm:

- `SimulationData.effective_norm` / `BEMResult.effective_norm` resolve
  `"auto"` → `acc` when the model is velocity-driven, `raw` otherwise, and
  refuse an outright `"acc"` that cannot be honoured.
- `SimulationData.is_velocity_driven` and `BEMResult.has_acc` expose the state.
- `sim.result()` carries `pressure_acc = None` for a scattering model rather
  than raising at construction, so the result is fully usable as raw.
- Every viz function defaults to `norm="auto"` and labels the axis from the
  *resolved* norm (`viz/_style.pressure_label`, `ir_label`), so a scattering
  plot shows Pa and says Pa.

Three `A_vib` bugs were fixed underneath, each of which quietly changed the
denominator:

- **Overlapping ranges double-counted.** A `[BOUNDARY]` entry may cover
  elements the vibrating group already drove; both emit a `VELO` line, and the
  triangle's area was added twice. Elements are now a set.
- **`VELO 0.0` counted as driven.** `v0 = 0` is exactly how NumCalc spells
  sound-hard, so an explicitly-rigid range inflated the area.
- **`pressure_at`/`velocity_at` skipped normalization silently** when `A_vib`
  was None (`norm == "acc" and self.A_vib is not None`), so the interactive 3D
  viewer drew raw data on an acc-labelled colour bar.

`PRES`/`ADMI`/`IMPE` elements are correctly *not* sources: only `VELO` lines
with a non-zero amplitude enter the vibrating area. That is what makes this tier
safe to land before Tier 2 rather than with it.
