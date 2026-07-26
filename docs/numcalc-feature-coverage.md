# NumCalc feature coverage — what bemkit exposes, what it doesn't

**Date:** 2026-07-26
**Source:** Kreuzer, Pollack, Brinkmann & Majdak, *NumCalc: An open-source BEM
code for solving acoustic scattering problems*, arXiv:2312.00005v3 (2024-10-31),
cross-checked against the NumCalc parser at
`mesh2hrtf/NumCalc/src/NC_Input.cpp`.

Where the paper and the source disagree, **the source wins** and the
disagreement is noted.

## Summary

Incident sources are not a missing feature — `bem` already writes plane waves
and point sources into `NC.inp` correctly, and NumCalc reads them. What is
missing is everything *around* them: typed configuration, validation against
the paper's hints, and the result-normalisation semantics that a
non-velocity-driven model needs.

No part of this requires architectural change. The BEM layer already has the
right shape: `[BOUNDARY]` selects elements (including by geometric anchor) and
`write_nc_inp` emits one line per range. Adding pressure/admittance conditions
is a change to *what value* is written, not to *how elements are chosen*. The
one change with a genuine cross-layer tail is normalisation, which lives in
`analysis/`, not `sim/`.

Rough sizing: four small, independent increments (Tiers 1, 2, 5 below), one
medium (Tier 3), and one that needs a design decision (Tier 4).

---

## 1. What NumCalc actually accepts

### 1.1 Boundary conditions (`NC_Input.cpp:1227-1290`)

```
ELEM e1 TO e2 <bctype> real(v0) curve1 imag(v0) curve2
```

| Keyword | `nbtyp` | Meaning |
|---|---|---|
| `VELO` | 0 | Particle velocity normal to the element. `0` ⇒ sound-hard (the default for any element with no BC line). |
| `PRES` | 1 | Sound pressure. `0` ⇒ sound-soft. |
| `ADMI` | 2 | Admittance (absorbing materials). |
| `IMPE` | 2 | Impedance. Converted internally to admittance: `Y = 1/Z`, i.e. `re/(re²+im²)`, `-im/(re²+im²)`. |
| `TRAD`, `TRIM` | 3 | **Parsed, then hard-error**: "Transfer admittance/impedance boundary condition is not supported!" Do not expose. |

Behaviours worth knowing, none of which are in the paper's prose:

- **`IMPE` with value `0+0i` silently becomes `PRES` sound-soft** (`:1268-1273`).
- **`IMPE` combined with a curve is a hard `exit(-1)`**, with the message
  "Please use ADMI conditions. They work with CURVES." So a frequency-dependent
  surface must be expressed as admittance, never impedance.
- **An element may carry both `ADMI` and `VELO`** — this is how a vibrating
  source on an absorbing surface is modelled (paper, Appendix A).
- A BC line needs **at least 9 whitespace-separated terms** (`:1232`).

### 1.2 Curve references are all-or-nothing (`NC_Input.cpp:1244-1250`)

```
ERROR: Curves have to be specified for all parts of a Boundary or Source.
I.e., ALL curve numbers in one line must be -1 or ALL curve numbers must
not be -1
```

This applies to **boundary conditions and sources alike**. A frequency-dependent
real part with a constant imaginary part is rejected. Any future `CURVES`
support must enforce the pairing at config-validation time rather than letting
NumCalc fail at solve time.

### 1.3 Incident sources (`NC_Input.cpp:1013-1068`)

```
PLANE WAVES
Nr  DX DY DZ  Real(P0) curve1 Imag(P0) curve2

POINT SOURCES
Nr  X0 Y0 Z0  Real(P0) curve1 Imag(P0) curve2
```

- Both require **exactly 8 terms**; anything else is a hard error.
- **The parser only compares the first token** — `chterms[0].compare("PLANE")`.
  So `PLANE` and `PLANE WAVES` are equally valid. The paper's Appendix C.4
  documents `POINT`/`PLANE` while its own example listing uses `PLANE WAVES`;
  both work, and **bemkit's current spelling is correct**.
- **For a plane wave, fields 1–3 are a direction, not a position**, and NumCalc
  **normalises it itself** (`Vcnorm_dim3_(dirinw[i], dw)`). The user need not
  supply a unit vector — but a zero vector is a division by zero.
- For a point source, fields 1–3 are a position in metres.

### 1.4 Main Parameters II is richer than we use (`NC_Input.cpp:292-322`)

```
nplanewaves  npointsources  isInternal  ClusEdgL0  IPRECOND
```

| Field | Variable | Notes |
|---|---|---|
| 3 | `ClusEdgL0_` | **FMM initial bounding-box edge length.** `0.0` ⇒ NumCalc's default, `(√N·A₀)^½`. |
| 4 | `methodPreconditioner_` | Must be in `[-3, 2]`. Negative values select incomplete-LU scanning degree: `-1`→1, `-2`→2, `≤-3`→3, each with `methodPreconditioner_=0`. Forced to `2` when the direct solver is used. |

Only these five fields are read; trailing values are ignored.

`ClusEdgL0_` is not a minor knob. The paper's Conclusions name shrinking the
initial bounding box as *the* remedy when the multipole expansion length `L`
blows up memory on large meshes (Sec. 3.3.1, Sec. 4.2, Conclusions), and
Sec. 4.2 says a large `L` also destabilises the solver.

### 1.5 Command-line (`Appendix D`)

- `NumCalc --estimate_ram` — runs **clustering only**, no BEM solve, and reports
  estimated memory. Slightly underestimates (ignores mesh/result arrays).
- `NumCalc -istart N -iend M` — already used by bemkit's job planner.

---

## 2. What bemkit does today

| NumCalc capability | bemkit | Where |
|---|---|---|
| `VELO`, real value | ✅ | `sim/writers.py:292` |
| `VELO`, complex value | ❌ imaginary part hardcoded `0.0` | same line |
| `PRES` | ❌ | — |
| `ADMI` / `IMPE` | ❌ | — |
| `ADMI` + `VELO` on one element | ❌ | — |
| `CURVES` (frequency-dependent) | ❌ curve refs hardcoded `-1` | `writers.py:292,305` |
| Plane waves | ⚠️ works, raw 8-value pass-through | `writers.py:165-180, 300-308` |
| Point sources | ⚠️ same | same |
| Element selection by geometry | ✅ and better than NumCalc needs | `sim/boundary.py` |
| `ClusEdgL0` | ❌ hardcoded `0.0000e+00` | `sim/templates/NC.inp:28` |
| Preconditioner | ❌ hardcoded `0` | same |
| `--estimate_ram` | ❌ | — |
| `-istart/-iend` | ✅ | `sim/runner.py:68` |
| Solver-convergence check (Hint 5) | ✅ | `sim/status.py:solver_warnings` |

Two accuracy problems in the current source support:

1. **The documentation mislabels plane-wave fields.** `writers.py:178` and
   `HOWTO.md` both describe the row as `[nr, x0, y0, z0, ...]`. For a plane wave
   those three numbers are a **direction**. A user copying the point-source
   example and changing the section name gets a plane wave travelling toward the
   origin-relative point they thought they were placing a source at.
2. **The `NC.inp` template comment is wrong.** "forth value is a NumCalc
   parameter (test 0.05 if issues)" describes `ClusEdgL0_`, the FMM bounding-box
   edge length. The advice is accidentally sound — 0.05 m shrinks the box — but
   it reads as folklore rather than the documented memory/stability knob it is.

---

## 3. The ten hints, mapped to bemkit

| # | Hint | bemkit status |
|---|---|---|
| 1 | Normal vectors point away from the scatterer | ❌ **not checked.** NumCalc does not check either, and the paper calls a flipped normal the most common error that still produces plausible-looking output. A watertight mesh admits a cheap global orientation test (signed volume via the divergence theorem). |
| 2 | 6–8 elements per wavelength | ⚠️ implicit. `[mesh] lc_max` is documented as "~6 elements per wavelength at max freq", but nothing verifies it against `sr`/`nbins` after meshing. `element_areas` already exists. |
| 3 | ≥ 1 average edge length between an external source and the surface | ❌ **not checked** — and this is the validation that matters most for the feature being added. bemkit already computes exactly this quantity for fieldpoints (`min_surface_distance_factor × avg_edge_length`); point sources need the same test. |
| 4 | Keep elements as regular as possible | ❌ not measured. Aspect-ratio/skew histogram would be cheap from `read_mesh_triangles`. |
| 5 | Check the iterative solver converged | ✅ `solver_warnings` scans `NC*.out` for both strings; surfaced in `bem status` and after a run. |
| 6 | Elements approximately the same size, at least locally | ❌ not measured. Directly relevant: the graded meshes bemkit produces (`lc_min`→`lc_max`) are exactly the case the hint warns about, and FMM stability depends on it. |
| 7 | Subdivision-loop error ⇒ overlaps, or a source too close | ❌ the error string is not in `_SOLVER_WARNINGS`. One-line addition to an existing mechanism. |
| 8 | Expansion order `L > 30` ⇒ check convergence and plausibility | ❌ `L` is reported per level in `NC.out` and is not parsed. |
| 9 | Check the plausibility of your results | — user's job, but sparse/partial results now make this easier. |
| — | Mesh must be watertight, each edge in exactly two elements | ❌ not checked. |

Hints 1, 3, 4, 6 and 7 are all *mesh or setup* diagnostics that bemkit is well
placed to provide and currently does not. They are independent of the source
feature and could ship as a `bem MODEL check` family (the CLI already reserves
that name for the private `bemlab` package — worth reconsidering).

---

## 4. Proposed increments

### Tier 1 — typed incident sources (small)

Replace the raw 8-value row with a declarative table, keeping the current form
working:

```toml
[sources.plane1]
type = "plane"
direction = [0, 0, 1]      # normalised by NumCalc; reject a zero vector
amplitude = 1.0            # or [re, im]

[sources.spk]
type = "point"
position = [0.0, -1.5, 0.0]   # metres
amplitude = 1.0
```

Work: a `sim/sources.py` mirroring `sim/boundary.py`'s shape (validate → emit
rows), config schema entries, and two validations — reject a zero direction
vector, and enforce Hint 3 by rejecting a point source closer than one average
edge length to the surface (reusing the mesh data `resolve_boundary` already
loads). Fix the mislabelled docs while there.

*No change to `write_nc_inp`'s emit loop; it already writes the rows.*

### Tier 2 — `PRES` / `ADMI` / `IMPE` boundary conditions (small)

The emit line becomes:

```python
nc.write(f'ELEM {s} TO {e} {bctype} {re:.4f} {c1} {im:.4f} {c2}\n')
```

Element *selection* needs nothing: `resolve_boundary` already returns ranges,
and geometric anchors work unchanged. What is needed is a per-entry type and a
complex value in the config schema:

```toml
[BOUNDARY.cap]
absorber = { near = [0, 0, 0.05], radius = 0.02, type = "admi", value = [0.003, -0.001] }
```

Plus: reject `TRAD`/`TRIM`, convert or reject `IMPE`+curve per §1.2, and allow
an element to carry both `ADMI` and `VELO` (two emitted lines for one entry).

### Tier 3 — `CURVES` (medium)

A new section writer after the sources, plus frequency-step→value curves. The
awkward part is not the format but the **coupling to the frequency grid**: curve
nodes are indexed by *frequency step*, so a curve is only meaningful alongside
the `sr`/`nbins` grid — and now also alongside partial-band runs, where only
some steps are computed. The all-or-nothing `-1` rule (§1.2) must be enforced at
config time.

### Tier 4 — normalisation semantics (small code, needs a decision)

`pressure_acc` divides by `ρ₀·ω·A_vib`, and `A_vib` comes from
`read_nc_inp_boundary_conditions`, which parses `VELO` lines **and only `VELO`
lines** (`sim/readers.py:135-192`). Consequences once Tier 2 lands:

- A `PRES`- or `ADMI`-only model gets `A_vib = None` and an unnormalised result
  with a warning that reads like a bug.
- A model mixing `VELO` and `ADMI` on the same elements would double-count area
  unless the reader distinguishes them.

The decision: does `norm="acc"` stay the default for models with no velocity
drive, or does the result layer learn to pick `raw` automatically and say so?
The same question already exists for pure scattering setups (documented in
`CLAUDE.md`), so Tier 4 is worth doing *with* Tier 1 rather than after it.

### Tier 5 — expose the two hardcoded solver knobs (trivial)

```toml
[bem]
cluster_edge_length = 0.05   # ClusEdgL0; 0 = NumCalc default
preconditioner = 0           # [-3, 2]
```

Two template fields and two validated config keys. Worth pairing with a
`bem MODEL estimate-ram` wrapper around `NumCalc --estimate_ram`, which answers
the question our `-j` default (1) currently answers by being conservative.

---

## 5. Answer to "would that require big modifications?"

**No.** Ranked by risk:

- **Tiers 1, 2 and 5 are additive** and touch one layer each. The reason they
  are cheap is that bemkit already separates *which elements* (`sim/boundary.py`,
  with geometric anchors NumCalc has no concept of) from *what is written*
  (`sim/writers.py`). Only the latter changes.
- **Tier 3 is self-contained** but couples to the frequency grid, which the
  partial-band work has just made more interesting.
- **Tier 4 is the only one that crosses into `analysis/`**, and it is a
  semantics decision, not a large diff. It is also the one that will silently
  produce wrong numbers if skipped — a `PRES`-driven model normalised as if it
  were velocity-driven is not obviously wrong on a plot.

The largest *genuine* gap the paper exposes is not the source or BC types at
all: it is that bemkit performs **none of the mesh-quality checks** the paper
spends ten hints on (§3), while NumCalc explicitly performs none either and
"will produce *some* results, and it is up to the user to check the validity".
That is the area where bemkit could add value that NumCalc does not.
