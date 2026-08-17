# sphcap_meshing — which meshing strategy actually buys accuracy

A 100 mm sphere with a ±30° vibrating cap, meshed nine ways, solved over
250 Hz – 16 kHz, and scored against the exact answer.

The spherical cap is one of the few radiation problems with a closed form that
is both exact and fast (`bemkit.model.analytic.sphcap_pressure`, a
spherical-harmonic series). That makes it an *instrument*: a mesh can be scored
against the truth rather than against a finer mesh, so "is this good enough"
stops being a convergence argument and becomes a number.

```bash
python models/sphcap_meshing/mesh_sphcap_meshing.py   # once; meshes are committed
bem models/sphcap_meshing check
bem models/sphcap_meshing solve -j 4
python models/sphcap_meshing/compare_analytic.py
```

**The caveat, stated up front:** this shape is everywhere smooth. Its only
discontinuity is the cap rim, where the velocity BC jumps from 1 to 0 — which
is what the `rim` variant is about, and as close to an edge as this geometry
gets. What a *geometric* edge needs is a separate study; nothing here answers it.

## Two errors, and why they are reported separately

Adding them together hides both.

**Geometry error.** Flat triangles inscribed in a sphere under-state the driven
cap's area. At low ka the radiated field is proportional to that area, so this
alone sets the error, it does not shrink with frequency, and no solver can get
under it. `compare_analytic.py` reports it as `cap area` and divides it out.

**Discretisation error.** What is left once the cap area is right — the actual
elements-per-wavelength question. It grows with ka, and it is the number to
read when choosing a mesh for a band.

They can partially cancel. A mesh whose *raw* error is lower than its
*corrected* error is not doing better; it is wrong in two directions at once,
and it will stop being lucky as soon as anything about the problem changes.
That is why both columns are printed.

## The variants

Every size is quoted as elements per wavelength at 16 kHz (λ = 21.25 mm at
c = 340 m/s). Each variant exists as a controlled comparison against another —
that is the point of the set, not its size.

| variant | strategy | elements |
|---|---|---|
| `u2` `u4` `u6` `u8` `u10` | uniform, 2 / 4 / 6 / 8 / 10 per wavelength | 754 … 16890 |
| `cap_far` | graded 8/λ on the cap → 2/λ on the far side | 3286 |
| `cap_mid` | graded 8/λ on the cap → 4/λ on the far side | 5716 |
| `geodesic` | same ramp as `cap_far`, measured over the surface | 3342 |
| `rim` | 16/λ band around the cap rim, 4/λ base | 4222 |

- **`u2`…`u10`** — the cost curve: what an element budget buys with no
  cleverness at all. Everything else is judged against it.
- **`cap_far` vs `u8`** (same cap size) and **vs `u2`** (same far-field size) —
  is grading towards the source worth it at ka ≈ 15?
- **`cap_mid` vs `cap_far`** — how coarse the far side can get before it bites.
- **`geodesic` vs `cap_far`** — negative control. The sphere is convex, so the
  two metrics rank identically and the meshes must come out the same. If they
  ever diverge here, `models/plate`'s geodesic result is suspect.
- **`rim` vs `u4`** (same base size) and **vs `u6`** (comparable count) — is the
  accuracy bought at the rim?

Two predictions made before solving, both wrong, which is most of what the
study was worth: `cap_far` was expected to track `u2` rather than `u8` (it
tracks `u8`, §1), and `rim` was expected to be the cheap win (it is the only
variant that loses to plain uniform refinement, §3).

## Results

Relative L2 error on the 1 m directivity, cap-area-corrected, at 16 kHz
(ka = 14.8). `cap` / `rigid` are elements per wavelength at 16 kHz on the
driven and passive parts, **area-weighted**; `cap p90` is the 90th-percentile
element size on the cap, i.e. its *coarsest* part.

| variant | tris | cap | cap p90 | rigid | cap area | err @16 kHz |
|---|---:|---:|---:|---:|---:|---:|
| u2 | 754 | 2.2 | 2.0 | 2.2 | −3.15% | 7.71% |
| u4 | 2804 | 4.3 | 4.0 | 4.2 | −0.80% | 1.91% |
| **rim** | 4222 | 5.8 | **4.7** | 4.2 | −0.18% | **2.14%** |
| u6 | 6188 | 6.4 | 6.1 | 6.2 | −0.35% | 0.98% |
| **cap_far** | **3286** | 8.4 | 7.9 | **3.3** | −0.20% | **0.67%** |
| **geodesic** | 3342 | 8.4 | 7.9 | 3.3 | −0.20% | 0.70% |
| **cap_mid** | 5716 | 8.4 | 7.9 | 5.4 | −0.20% | 0.63% |
| u8 | 11056 | 8.4 | 7.9 | 8.3 | −0.20% | 0.63% |
| u10 | 16890 | 10.5 | 10.1 | 10.2 | −0.13% | 0.47% |

### 1. Resolve the driven patch. The passive surface barely matters.

`cap_far` reaches `u8`'s accuracy with **30% of its elements**. Its rigid side
runs at 3.3 elements per wavelength — half the usual rule of thumb — and costs
essentially nothing, because that rule is about resolving the *field*, and on
the shadow side of a driven cap the field is small.

Elements a uniform mesh would need for the same accuracy:

| | 1 kHz | 4 kHz | 16 kHz |
|---|---:|---:|---:|
| `cap_far` (3286 tris) | 4.2× | 1.5× | **3.1×** |
| `cap_mid` (5716 tris) | 3.0× | 1.8× | 1.9× |
| `rim` (4222 tris) | 1.9× | **0.4×** | **0.6×** |

The saving is real across the band and largest at the top of it. The dip at
4 kHz is where the two error mechanisms cross over (see §6).

### 2. What limits the answer is the *coarsest* element on the cap.

Sort the table by `cap p90` and the error falls almost monotonically, while
`rigid` ranges over 3.3–10.2 with no visible effect. The mean is the wrong
statistic: a count-weighted mean over a graded mesh is dominated by its many
small elements and flatters it — `rim`'s mean cap size is 1.98 mm, *finer than
u10's*, while the interior it actually has to resolve sits at 5.31 mm.

### 3. Local rim refinement is a trap.

`rim` has the best geometry of any variant (cap area −0.18%) and by far the
finest rim (1.70 mm against `cap_far`'s 2.54), spends 40% of its elements
within 8 mm of the cap edge — and is **the worst strategy tested**. At 16 kHz
it is worse than `u4`, which has 50% fewer elements. A uniform mesh of 60% its
size would do better.

Not element quality: `rim` and `cap_mid` have identical `bem check` uniformity
statistics (neighbour area ratio median 1.08, max 3.01) and `cap_mid` does
fine. It is §2 — `rim` left the cap *interior* at 4/λ, so it inherited `u4`'s
error and paid extra for a rim that was never the limiter.

The surface-pressure split (§4) shows this from the other side: `rim` is the
one mesh that *does* flatten the rim's local error, and its radiated field is
still the worst of the strategies.

### 4. The rim carries ~2× the surface error — and fixing that does not help.

Boundary pressure against the closed form at 16 kHz, split by distance to the
cap edge:

| variant | tris | all | at rim | away | ratio |
|---|---:|---:|---:|---:|---:|
| u2 | 754 | 11.73% | 13.94% | 8.22% | 1.70× |
| u4 | 2804 | 4.24% | 5.34% | 2.22% | 2.41× |
| u6 | 6188 | 2.38% | 3.02% | 1.36% | 2.21× |
| u8 | 11056 | 1.65% | 2.09% | 0.97% | 2.17× |
| u10 | 16890 | 1.19% | 1.50% | 0.71% | 2.11× |
| **rim** | 4222 | 3.10% | 3.18% | 2.57% | **1.24×** |

The rim band really does carry about twice the local error of the rest of the
surface, steadily across the uniform series. But both regions converge at
roughly the same rate (~1.5 between u8 and u10, matching the far field), so
the rim is a **uniformly elevated region, not a convergence bottleneck**.

`rim` is the test: it is the one mesh that flattens the disparity to 1.24×,
doing exactly what it was designed to do on the surface — and its radiated
field is still the worst of the strategies (§3). Local surface error at the rim
is simply not what the far field is limited by.

> **Correction.** An earlier version of this file reported 6.30% / 2.37% here
> and claimed the rim error "stops falling with refinement". Both were wrong:
> the reference series was truncated at the auto-selected `nmax = 35`, which is
> converged to 1e-12 at a field point but **not on the surface**, where the
> modal filter decays like 1/n instead of superexponentially. That overstated
> the boundary error about fourfold and worst of all at the rim — manufacturing
> the plateau. The table above uses `nmax = 200`. Field-point results were never
> affected: they are identical to four decimals from nmax 35 to 250.

`analytic_truncation.png` is the diagnostic that caught this — the closed form
scored against itself, with no mesh involved. At a field point the modal filter
`h_n(kr)/h_n′(ka)` decays like `(a/r)ⁿ`, so the series falls off a cliff past
n ≈ ka and is at 1e-13 by nmax 45. On the surface that filter decays like `1/n`
and the cap's own spectrum like `1/n`; the curve is a straight line on log-log,
still worth ~0.4% at nmax 200. The right panel is the other half of the trap:
the order is bounded *above* too, since `y_n` overflows past roughly n = 5ka+20,
and at 250 Hz the whole usable window is nmax 21…118.

Two limits travel with this metric and are not removable. The series has a
Gibbs floor at the velocity discontinuity worth about 0.02 pp, which no order
removes. And the analytic value is taken at element centroids projected
radially onto r = R, since a flat triangle's centroid lies inside the sphere
where the exterior series does not apply — an O(h²) inconsistency of the same
order as the quantity being measured. It localises; it does not carry an
absolute accuracy claim.

### 5. Geodesic ≡ euclidean on a convex body — the control passes.

1.7% apart in element count and 0.03 pp apart in error. That is what it should
be: the sphere is convex, so the two metrics rank every point identically. The
`models/plate` result, where they differ sharply, is therefore about the
geometry folding back on itself and not about the algorithm.

### 6. Two errors, opposite signs, cancelling near 3 kHz.

The raw (uncorrected) curves dip around 3 kHz and rise on either side: below
it the cap-area deficit dominates, above it discretisation does, and they have
opposite sign. `u2`'s raw error at 4 kHz is 2.41% against a corrected 2.29%,
but at 250 Hz it is 3.06% raw against 0.42% corrected. **A mesh validated at a
single mid-band frequency will flatter itself.**

### 7. Uniform refinement does not hold second order.

The observed order between successive uniform steps at 16 kHz falls
**2.12 → 1.62 → 1.68 → 1.33** rather than holding at 2.00. That is measured on
the 1 m directivity, where the reference is exact to 1e-12, so the decline is
real.

The obvious culprit is the cap rim's BC discontinuity capping the global rate,
which is textbook. This study does not establish it: §4 finds the rim and the
rest of the surface converging at *similar* rates, which is not what a
rim-limited rate looks like. Adding more uniform elements has clearly
diminishing returns here; **why** is not settled by these nine meshes. Either
way it is the reason §1's answer is "move the elements onto the driven patch"
rather than "use more of them".

### 8. No irregular frequencies anywhere.

Across ka = 0.23…14.8 — dozens of interior Dirichlet eigenvalues — not one
isolated error spike on any of the nine meshes. NumCalc's Burton-Miller
coupling works, so an exterior sweep needs no CHIEF points and no
band-splitting. (`BemppSolver` is the one that must stay below ka = π.)

### 9. `bem check`'s elements-per-wavelength test misjudges a graded mesh.

It rejects `cap_far` and `geodesic` outright — "1.2 elements per wavelength at
16000 Hz … the top of the band is not resolved" — for the two meshes that turn
out to be the most efficient of the nine. It measures the **globally longest
edge**, so a mesh that deliberately puts its coarse elements where the field is
weak fails on the strength of its cheapest region.

Not wrong for a general mesh, where nothing says which part is driven, and it
is right about every uniform variant. But on this study it would have steered a
user away from a mesh that beats the "passing" one by 3× in elements. A
per-group version of the check — the driven group against λ, the passive
group looser — would say something useful here. Left as it is for now; the
observation is the point.

### What this does not answer

Everything above is about a **smooth** surface whose only discontinuity is in
the boundary condition. §3's "don't refine the feature, refine the patch" is a
result about a *BC* jump; a geometric edge is a different object — the field
itself is singular there, not just its surface trace — and nothing here says
what it needs. That study wants a shape with a real edge (a finite disc, a
cylinder, a wedge) and, ideally, another closed form. `edge_field` already
takes an explicit `curves=` list for exactly that case.

## Notes for whoever runs this next

- **NumCalc is Burton–Miller.** `NC_EquationSystem.cpp:85` sets the coupling
  constant `zBta3 = i·τ/k` for exterior problems (0 for interior). This sweep
  crosses dozens of interior Dirichlet eigenvalues (the first is ka = π,
  3.4 kHz) and `compare_analytic.py`'s spike check finds no isolated error peak
  on any variant — so a full-band sphere sweep needs no CHIEF points and no
  band-splitting. `BemppSolver` is the one that does have to stay below ka = π.
- **The geodesic pass warns about one unreachable node.** It is the degenerate
  vertex at the south pole, which belongs to no triangle in the pilot mesh, so
  Dijkstra never reaches it. Harmless — it is sized at `lc_max`, which is what
  the far pole gets anyway — but the warning is real and worth not chasing.
- **`edge_field` switches gmsh's boundary size extension off**, unlike the
  face-distance builders. It has to: gmsh seeds a surface from its bounding
  curves, and those are the very curves this field measures from, so `lc_min`
  otherwise spreads across the whole sphere. Measured here, leaving it on put
  the far side at 3.5 mm instead of 5.3 and more than doubled `rim`'s element
  count — a "rim-refined" mesh that was really just a fine mesh.
- **`models/sphcap` is a different study.** It is the 20° fixture that
  `tests/solvers/test_cross_validation.py` and three other test files pin. This
  one is free to change; that one is not.
