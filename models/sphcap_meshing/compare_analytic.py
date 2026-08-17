#!/usr/bin/env python3
"""Score every mesh variant against the exact spherical-cap solution.

    python models/sphcap_meshing/compare_analytic.py
    python models/sphcap_meshing/compare_analytic.py --no-plot

Two errors are separated here, because they behave completely differently and
adding them together hides both.

**Geometry error.** Flat triangles inscribed in a sphere under-state the driven
cap's area. At low ka the radiated field is proportional to that area, so this
alone sets the error, it does not shrink with frequency, and no solver can get
under it. It is reported as `cap area` and divided out to give the corrected
columns.

**Discretisation error.** What is left once the cap area is right: the actual
elements-per-wavelength question. It grows with ka, and it is the number to
read when choosing a mesh for a band.

They can also partially cancel — a coarse mesh whose raw error is *lower* than
its corrected error is not doing better, it is wrong in two directions at once.
That is why both columns are printed.

Needs the `view` extra for the figures; `--no-plot` needs only numpy + scipy.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from bemkit.model.analytic import sphcap_pressure       # noqa: E402
from bemkit.results import Results                      # noqa: E402
from bemkit.study import load_study, workdir            # noqa: E402

from model import ANGLE_DEG, RADIUS_MM, VARIANTS        # noqa: E402

#: Frequencies the table reports at, Hz. Bin-aligned (the band is 250 Hz
#: steps), spread over three decades of ka from 0.23 to 14.8.
REPORT_HZ = (250.0, 1000.0, 4000.0, 8000.0, 16000.0)

#: The field-point set the headline numbers come from: full directivity at 1 m.
PRIMARY = "polar"

#: The uniform family is an ordered resolution series and gets an ordinal blue
#: ramp; the strategies get categorical hues (orange/aqua/violet — validated
#: all-pairs in light mode by the dataviz palette validator, and clear of the
#: ramp's blue, which the uniform family owns).
UNIFORM = ("u2", "u4", "u6", "u8", "u10")
STRATEGY = ("cap_far", "cap_mid", "rim")
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b")
HUES = ("#eb6834", "#1baf7a", "#4a3aa7")

#: geodesic is not a fourth identity — it is cap_far's replicate under a
#: different distance metric, and the finding is that the two coincide. Giving
#: it its own hue would claim otherwise (and would land a fourth categorical
#: slot that does not validate against these three); it is drawn as an open
#: ring in cap_far's colour instead, which is what a replicate looks like.
REPLICATE = ("geodesic", "cap_far")

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"

EXACT_CAP_MM2 = (2 * math.pi * RADIUS_MM ** 2
                 * (1 - math.cos(math.radians(ANGLE_DEG))))


def element_areas_mm2(mesh):
    v = mesh.vertices[mesh.triangles]
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) * 1e6


def relative_l2(got, want):
    """Per-frequency ``||got - want|| / ||want||`` over the point axis.

    An L2 norm over the whole directivity rather than a max over points: a
    single near-null point where |want| is small would otherwise dominate a
    pointwise relative error and say nothing about the pattern.
    """
    return (np.linalg.norm(got - want, axis=0)
            / np.linalg.norm(want, axis=0))


#: Elements this far (mm) from the cap rim count as "at the rim". Two rim-band
#: element widths on the coarsest variant, so the split is about the feature
#: rather than about any one mesh's resolution.
RIM_BAND_MM = 8.0

#: Truncation order for the *boundary* comparison, which needs its own and a
#: much higher one than the field points do.
#:
#: At a field point the modal filter ``h_n(kr) / h_n'(ka)`` kills orders past
#: ``ka`` superexponentially, so the series is at machine precision by nmax 45
#: and the auto-selected 35 is already converged to 1e-12. On the surface
#: ``r = a`` that filter decays only like ``1/n``, and the cap's own spectrum
#: only like ``1/n``, so the surface series converges *algebraically*.
#: Measured on u10 at 16 kHz, the auto order overstates the boundary error
#: fourfold — 4.84 % against 1.19 % converged — and overstates it worst exactly
#: at the rim, which is where the interesting claim lives. Increments fall
#: below 0.05 pp by 200 and then sit on a ~0.02 pp floor: that floor is Gibbs
#: ringing at the velocity discontinuity, which no order removes, and it is the
#: accuracy limit of this metric.
#:
#: 200 is safe only well above ``ka ~ 1`` (``y_n`` overflows past roughly
#: n = 5ka + 20 — at 250 Hz the ceiling is 115), which is the other reason this
#: metric is reported at the top of the band only.
BOUNDARY_NMAX = 200
BOUNDARY_HZ = 16000.0


def rim_distance_mm(mesh):
    """Distance from each element centroid to the cap rim circle, mm."""
    c = mesh.vertices[mesh.triangles].mean(axis=1) * 1e3
    theta = math.radians(ANGLE_DEG)
    return np.hypot(np.hypot(c[:, 0], c[:, 1]) - RADIUS_MM * math.sin(theta),
                    c[:, 2] - RADIUS_MM * math.cos(theta))


def surface_error(model, results):
    """Boundary-pressure error at BOUNDARY_HZ, split by distance to the rim.

    A secondary metric with two limits of its own, both of which have to be
    carried alongside its numbers rather than discovered later:

    1. The series converges only algebraically here and hits a Gibbs floor at
       the velocity discontinuity — see BOUNDARY_NMAX. Worth about 0.02 pp.
    2. The analytic value is taken at each element centroid *projected radially
       out to r = R*. Centroids of flat triangles sit inside the sphere, where
       the exterior series does not apply, so a projection is unavoidable; it
       is an O(h^2) inconsistency of the same order as the thing being
       measured.

    So this localises where surface error sits, and does not carry an absolute
    accuracy claim. The field-point columns are the headline numbers.
    """
    mesh = model.mesh
    centroids = mesh.vertices[mesh.triangles].mean(axis=1)
    on_sphere = (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
                 * (RADIUS_MM / 1e3))
    column = [int(np.argmin(np.abs(results.frequencies - BOUNDARY_HZ)))]
    want = sphcap_pressure(on_sphere, results.frequencies[column],
                           c0=model.medium.c, rho0=model.medium.rho,
                           angle_deg=ANGLE_DEG, radius_mm=RADIUS_MM,
                           nmax=BOUNDARY_NMAX)
    got = results.boundary_pressure()[:, column]
    at_rim = rim_distance_mm(mesh) < RIM_BAND_MM
    return {"all": relative_l2(got, want),
            "rim": relative_l2(got[at_rim], want[at_rim]),
            "away": relative_l2(got[~at_rim], want[~at_rim]),
            "rim_share": at_rim.mean()}


def spikes(error, factor=3.0):
    """Bins whose error stands out from their neighbours by `factor`.

    The irregular-frequency check. A direct exterior formulation breaks down at
    the interior Dirichlet eigenvalues (for a 50 mm sphere the first is
    ka = pi, 3.4 kHz, and there are dozens more below 16 kHz), and the symptom
    is an isolated spike, not a trend. NumCalc applies Burton-Miller coupling
    (NC_EquationSystem.cpp:85), so there should be none — this is what checks
    that rather than assuming it.
    """
    if len(error) < 5:
        return []
    pad = np.pad(error, 2, mode="edge")
    local = np.array([np.median(np.delete(pad[i:i + 5], 2))
                      for i in range(len(error))])
    return list(np.flatnonzero(error > factor * local))


def measure(study_dir, name, model):
    """Everything the report knows about one variant, or None if unsolved."""
    npz = workdir(study_dir, name) / "results.npz"
    if not npz.exists():
        return None
    results = Results.load(npz)

    areas = element_areas_mm2(model.mesh)
    h = np.sqrt(4 * areas / np.sqrt(3))
    cap_ratio = areas[model.mesh.groups["vibrating"]].sum() / EXACT_CAP_MM2
    at_rim = rim_distance_mm(model.mesh) < RIM_BAND_MM
    cap, rigid = (model.mesh.groups["vibrating"], model.mesh.groups["rigid"])

    errors = {}
    for field, points in model.fields.items():
        want = sphcap_pressure(points, results.frequencies,
                               c0=model.medium.c, rho0=model.medium.rho,
                               angle_deg=ANGLE_DEG, radius_mm=RADIUS_MM)
        got = results.pressure(field)
        errors[field] = {"raw": relative_l2(got, want),
                         "corrected": relative_l2(got / cap_ratio, want)}

    return {"name": name, "tris": len(model.mesh.triangles),
            "h_cap": np.average(h[cap], weights=areas[cap]),
            "h_cap_p90": np.percentile(h[cap], 90),
            "h_rigid": np.average(h[rigid], weights=areas[rigid]),
            "h_rim": h[at_rim].mean(), "cap_ratio": cap_ratio,
            "frequencies": results.frequencies, "errors": errors,
            "surface": surface_error(model, results),
            "solved": int(np.isfinite(results.pressure(PRIMARY)[0]).sum())}


def at(m, hz, field=PRIMARY, kind="corrected"):
    i = int(np.argmin(np.abs(m["frequencies"] - hz)))
    return m["errors"][field][kind][i]


# --- reporting ---------------------------------------------------------------

#: Wavelength at the top of the band, mm — what every element size is quoted
#: against (c = 340 m/s, see model.py).
LAMBDA_MIN_MM = 340.0 / 16000.0 * 1e3


def print_table(rows):
    head = (f"{'variant':<9} {'tris':>6} {'cap':>5} {'rigid':>6} "
            f"{'cap area':>9}  "
            + "  ".join(f"{hz / 1000:g}k" .rjust(13) for hz in REPORT_HZ))
    print("\nRelative L2 error on '%s' — raw / cap-area-corrected\n" % PRIMARY)
    print(head)
    print("-" * len(head))
    for m in rows:
        cells = "  ".join(
            f"{at(m, hz, kind='raw') * 100:5.2f}/{at(m, hz) * 100:<6.2f}"
            .rjust(13) for hz in REPORT_HZ)
        print(f"{m['name']:<9} {m['tris']:>6d} "
              f"{LAMBDA_MIN_MM / m['h_cap']:>5.1f} "
              f"{LAMBDA_MIN_MM / m['h_rigid']:>6.1f} "
              f"{(m['cap_ratio'] - 1) * 100:>+8.2f}%  {cells}")
    print("\n('cap' and 'rigid' are elements per wavelength at 16 kHz on the "
          "driven and passive\n parts, **area-weighted**: a plain mean counts "
          "elements, so a graded mesh's many\n small ones hide however coarse "
          "the rest of it is.\n cap area is the meshed driven area against the "
          "exact spherical cap; the corrected\n column divides it out, leaving "
          "BEM discretisation error alone.)")


def print_secondary(rows):
    print(f"\nAt 16 kHz, %        cap coarsest    mesh at rim     field points"
          f"      surface pressure")
    print(f"{'variant':<9} {'p90 h':>6} {'el/wl':>6}  {'h mm':>5} {'els':>6}"
          f"   {'near':>6} {'onaxis':>7}   {'all':>6} {'at rim':>7} {'away':>6}")
    print("-" * 82)
    for m in rows:
        s = m["surface"]
        print(f"{m['name']:<9} {m['h_cap_p90']:>6.2f} "
              f"{LAMBDA_MIN_MM / m['h_cap_p90']:>6.1f}  "
              f"{m['h_rim']:>5.2f} {s['rim_share'] * 100:>5.1f}%"
              f"   {at(m, 16000, 'near') * 100:>6.2f} "
              f"{at(m, 16000, 'onaxis') * 100:>7.2f}   "
              f"{s['all'][-1] * 100:>6.2f} {s['rim'][-1] * 100:>7.2f} "
              f"{s['away'][-1] * 100:>6.2f}")
    print(f"\n('cap coarsest' is the 90th percentile element size on the driven "
          f"cap — what the\n answer is actually limited by, as the `rim` "
          f"variant shows. 'mesh at rim' describes\n the mesh within "
          f"{RIM_BAND_MM:g} mm of the cap edge. Field columns are "
          f"cap-area-corrected.\n 'surface' is boundary pressure at "
          f"radially projected centroids, reference truncated\n at "
          f"nmax={BOUNDARY_NMAX} — a localiser carrying ~0.02 pp of Gibbs floor "
          f"and an O(h^2) projection\n inconsistency of its own; not comparable "
          f"to the field columns. See surface_error.)")


def print_checks(rows):
    print("\nChecks")
    print("-" * 60)
    bad = [(m["name"], m["frequencies"][i])
           for m in rows for i in spikes(m["errors"][PRIMARY]["corrected"])]
    if bad:
        print("  ! irregular-frequency spikes: "
              + ", ".join(f"{n} @ {f:.0f} Hz" for n, f in bad))
    else:
        print(f"  ok  no error spike on any variant across "
              f"{len(rows[0]['frequencies'])} bins — Burton-Miller holds")

    partial = [m["name"] for m in rows if m["solved"] < len(m["frequencies"])]
    if partial:
        print(f"  !   still solving (partial bins): {', '.join(partial)}")

    have = {m["name"]: m for m in rows}
    uni = [have[n] for n in UNIFORM if n in have]
    if len(uni) > 1:
        errs = [at(m, 16000) for m in uni]
        monotone = all(a > b for a, b in zip(errs, errs[1:]))
        print(f"  {'ok ' if monotone else '!  '} uniform series at 16 kHz "
              f"falls monotonically: "
              + " > ".join(f"{e * 100:.2f}" for e in errs) + " %")
        # The observed order, not an assumed one. Uniform refinement against a
        # boundary-condition discontinuity is capped below h^2 no matter how
        # fine the rest gets -- a declining order here is the expected shape,
        # and a flat *error* would be the thing to worry about.
        orders = [math.log(a / b) / math.log(m1["h_cap"] / m2["h_cap"])
                  for a, b, m1, m2 in zip(errs, errs[1:], uni, uni[1:])]
        print(f"      observed order between steps: "
              + " -> ".join(f"{p:.2f}" for p in orders)
              + "  (h^2 would be 2.00 throughout)")

    if {"cap_far", "geodesic"} <= set(have):
        d = abs(at(have["geodesic"], 16000) - at(have["cap_far"], 16000)) * 100
        n = abs(have["geodesic"]["tris"] / have["cap_far"]["tris"] - 1) * 100
        verdict = "ok " if d < 0.2 and n < 10 else "!  "
        print(f"  {verdict} geodesic vs cap_far on a convex body: "
              f"{n:.1f}% element count, {d:.2f} pp error (gate: <10%, <0.2pp)")


# --- figures -----------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color="#e8e7e2", lw=0.8, zorder=0)
    ax.grid(True, which="minor", color="#f2f1ed", lw=0.6, zorder=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(INK_3)
    ax.tick_params(which="both", colors=INK_2, labelsize=8, length=3)


def _si(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def _log_ticks(ax, which, ticks, fmt=_si):
    """Readable ticks on a log axis.

    Matplotlib's default puts a scientific-notation label on every *minor*
    decade subdivision of a log axis that spans less than a decade or two,
    which on these ranges collides into an unreadable smear. Naming the ticks
    is both legible and more informative — the reader wants 16k, not 10^4.2.
    """
    import matplotlib.ticker as mticker

    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt(v)))
    axis.set_minor_locator(mticker.NullLocator())


def _label_last(ax, x, y, text, color, dy=0):
    """Direct label at the right end of a trace.

    Not decoration: the aqua slot sits under 3:1 against the chart surface,
    which the palette's relief rule answers with visible labels rather than a
    legend swatch alone. `dy` staggers labels whose traces converge — which
    the graded variants do, that being the result.
    """
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, dy),
                textcoords="offset points", va="center", fontsize=7.5,
                color=color)


#: Vertical label offsets (points) where traces land on top of each other at
#: the right edge. cap_far / cap_mid / geodesic all converge near 0.65 % at
#: 16 kHz, which is exactly what the study found and exactly what makes the
#: default labelling unreadable.
LABEL_DY = {"cap_far": 9, "cap_mid": -9, "geodesic": 22}


def plot_frequency(rows, path):
    import matplotlib.pyplot as plt

    have = {m["name"]: m for m in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True,
                             facecolor=SURFACE)
    replicate, of_which = REPLICATE
    families = ((UNIFORM, RAMP, "uniform, 2-10 per wavelength"),
                (STRATEGY, HUES, "graded and rim-refined"))
    for row, (names, colors, family) in enumerate(families):
        for col, kind in enumerate(("raw", "cap-area-corrected")):
            ax = axes[row][col]
            _style(ax)
            key = "raw" if kind == "raw" else "corrected"
            for name, color in zip(names, colors):
                m = have.get(name)
                if m is None:
                    continue
                f, e = m["frequencies"], m["errors"][PRIMARY][key] * 100
                ax.plot(f, e, color=color, lw=2.0, zorder=3,
                        solid_capstyle="round")
                _label_last(ax, f, e, name, color, LABEL_DY.get(name, 0))
            m = have.get(replicate) if row == 1 else None
            if m is not None:
                color = HUES[STRATEGY.index(of_which)]
                f, e = m["frequencies"], m["errors"][PRIMARY][key] * 100
                ax.plot(f, e, color=color, lw=1.0, ls=(0, (2, 2)), zorder=4)
                _label_last(ax, f, e, f"{replicate} (= {of_which})", color,
                            LABEL_DY.get(replicate, 0))
            ax.set_xscale("log")
            ax.set_yscale("log")
            _log_ticks(ax, "x", [250, 500, 1000, 2000, 4000, 8000, 16000])
            _log_ticks(ax, "y", [0.1, 0.2, 0.5, 1, 2, 5, 10], fmt=lambda v: f"{v:g}")
            ax.set_xlim(230, 21000)
            ax.set_title(f"{family}  —  {kind}",
                         color=INK, fontsize=10, loc="left", pad=8)
            if col == 0:
                ax.set_ylabel("relative L2 error, %", color=INK_2, fontsize=8.5)
            if row == 1:
                ax.set_xlabel("frequency, Hz", color=INK_2, fontsize=8.5)

    fig.suptitle("Spherical cap: BEM against the closed form, 250 Hz - 16 kHz",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.945,
             "left: as solved.  right: divided by the meshed cap area, which "
             "removes the geometry error and leaves discretisation alone.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_cost(rows, path, frequencies=(1000.0, 4000.0, 16000.0)):
    """Error against element count: is a strategy worth its elements?

    The uniform series is the cost curve — what a given element budget buys
    with no cleverness at all. A strategy is only worth using if its point
    lands *below* that line at its own element count.
    """
    import matplotlib.pyplot as plt

    have = {m["name"]: m for m in rows}
    fig, axes = plt.subplots(1, len(frequencies), figsize=(12, 4.2),
                             sharey=True, facecolor=SURFACE)
    for ax, hz in zip(np.atleast_1d(axes), frequencies):
        _style(ax)
        uni = [have[n] for n in UNIFORM if n in have]
        if uni:
            x = [m["tris"] for m in uni]
            y = [at(m, hz) * 100 for m in uni]
            ax.plot(x, y, color=RAMP[3], lw=2.0, zorder=3, marker="o",
                    ms=6, mfc=RAMP[3], mec=SURFACE, mew=1.5)
            _label_last(ax, x, y, "uniform", RAMP[3])
            # Slope -1: error goes as h^2 and elements as h^-2, so a mesher
            # that only makes elements smaller must ride this line.
            guide = np.array([min(x) * 0.85, max(x) * 1.2])
            ax.plot(guide, y[0] * (guide / x[0]) ** -1.0, color=INK_3,
                    lw=1.0, ls=(0, (4, 3)), zorder=2)
        offsets = {"cap_far": (0, 10), "cap_mid": (0, -18), "rim": (0, 10)}
        for name, color in zip(STRATEGY, HUES):
            m = have.get(name)
            if m is None:
                continue
            ax.plot([m["tris"]], [at(m, hz) * 100], marker="D", ms=7,
                    color=color, mec=SURFACE, mew=1.5, zorder=4, ls="none")
            ax.annotate(name, (m["tris"], at(m, hz) * 100),
                        xytext=offsets.get(name, (0, 10)),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=color)
        # The replicate lands on top of its original — which is the result, so
        # it is drawn as a ring around it rather than as a fourth identity.
        replicate, of_which = REPLICATE
        m = have.get(replicate)
        if m is not None:
            ax.plot([m["tris"]], [at(m, hz) * 100], marker="o", ms=13,
                    mfc="none", mec=HUES[STRATEGY.index(of_which)], mew=1.4,
                    zorder=5, ls="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_ticks(ax, "x", [500, 1000, 2000, 5000, 10000, 20000])
        _log_ticks(ax, "y", [0.1, 0.2, 0.5, 1, 2, 5, 10], fmt=lambda v: f"{v:g}")
        ax.set_title(f"{hz / 1000:g} kHz   (ka = {2 * np.pi * hz / 340 * 0.05:.1f})",
                     color=INK, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("elements", color=INK_2, fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel(
        "corrected relative L2 error, %", color=INK_2, fontsize=8.5)

    fig.suptitle("What an element budget buys", color=INK, fontsize=12,
                 x=0.02, ha="left")
    fig.text(0.02, 0.90, "dashed: slope -1, what pure refinement gives. A "
                         "strategy earns its place only below the uniform "
                         "line. Ring: geodesic, cap_far's replicate.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def truncation_study(frequency=BOUNDARY_HZ, radius_mm=RADIUS_MM,
                     angle_deg=ANGLE_DEG, c0=340.0, rho0=1.2):
    """How wrong the *reference* is, as a function of its own truncation order.

    Independent of any mesh: the evaluation sets are analytic point sets, so
    this measures the closed form against itself and nothing else.

    Three radii, because the answer differs qualitatively between them. At a
    field point the modal filter ``h_n(kr) / h_n'(ka)`` decays like
    ``(a/r)^n`` — exponential, and the series is exact long before the
    truncation the study uses. On the surface ``r = a`` that filter decays like
    ``1/n`` and the cap's own spectrum like ``1/n``, so convergence is
    algebraic and stalls on a Gibbs floor at the velocity discontinuity.
    """
    a = radius_mm / 1e3
    theta = math.radians(angle_deg)
    f = np.array([float(frequency)])

    # A meridian arc of surface points, and two field circles. Colatitude is
    # what the cap edge is defined in, so the rim split is exact here rather
    # than approximated from element centroids.
    colat = np.radians(np.linspace(0.0, 180.0, 721))
    arc = np.stack([np.sin(colat), np.zeros_like(colat), np.cos(colat)], 1)
    near_rim = np.abs(np.degrees(colat) - angle_deg) < np.degrees(
        RIM_BAND_MM / radius_mm)

    sets = {
        "field, r = 1 m":       (arc * 1.0, None),
        "field, r = 75 mm":     (arc * 0.075, None),
        "surface, away from rim": (arc * a, ~near_rim),
        "surface, at rim":      (arc * a, near_rim),
    }

    orders = np.unique(np.round(np.geomspace(4, 200, 26)).astype(int))
    top = max(int(orders[-1]) * 1.25, 250)
    curves = {}
    for label, (points, mask) in sets.items():
        ref = sphcap_pressure(points, f, c0, rho0, angle_deg, radius_mm,
                              nmax=int(top))[:, 0]
        got = []
        for n in orders:
            p = sphcap_pressure(points, f, c0, rho0, angle_deg, radius_mm,
                                nmax=int(n))[:, 0]
            sel = slice(None) if mask is None else mask
            got.append(float(np.linalg.norm(p[sel] - ref[sel])
                             / np.linalg.norm(ref[sel])))
        curves[label] = np.array(got)
    return orders, curves, int(top)


def representable_ceiling(frequencies, radius_mm=RADIUS_MM, c0=340.0):
    """Largest nmax the Hankel recursion survives, per frequency.

    Not a convergence limit — a floating-point one. ``y_n(x)`` overflows for n
    well past x, so the usable window is bounded above as well as below, and at
    the bottom of the band it is narrow.
    """
    out = []
    for hz in frequencies:
        lo, hi = 4, 600
        while lo < hi:                      # last order that does not raise
            mid = (lo + hi + 1) // 2
            try:
                sphcap_pressure(np.array([[0.0, 0.0, radius_mm / 1e3]]),
                                np.array([float(hz)]), c0, 1.2, ANGLE_DEG,
                                radius_mm, nmax=mid)
                lo = mid
            except ValueError:
                hi = mid - 1
        out.append(lo)
    return np.array(out)


def plot_truncation(path, frequency=BOUNDARY_HZ):
    import matplotlib.pyplot as plt

    orders, curves, top = truncation_study(frequency=frequency)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                 facecolor=SURFACE, width_ratios=(1.15, 1))

    _style(ax)
    styles = {"field, r = 1 m":         (RAMP[3], "-"),
              "field, r = 75 mm":       (RAMP[1], "-"),
              "surface, away from rim": (HUES[1], "-"),
              "surface, at rim":        (HUES[0], "-")}
    floor = 1e-13
    # Label each curve at the last point still on the canvas: the field-point
    # traces fall off the bottom of the axis long before the right edge, so a
    # right-edge label would sit on empty space or vanish entirely.
    dy = {"field, r = 1 m": 10, "field, r = 75 mm": -14,
          "surface, at rim": 11, "surface, away from rim": -14}
    for label, err in curves.items():
        color, ls = styles[label]
        y = np.maximum(err * 100, floor)
        ax.plot(orders, y, color=color, ls=ls, lw=2.0, zorder=3,
                solid_capstyle="round")
        visible = np.flatnonzero(y > floor * 3)
        i = int(visible[-1]) if len(visible) else len(y) - 1
        ax.annotate(label, (orders[i], y[i]), xytext=(-4, dy[label]),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=color)
    for n, note in ((35, "auto (kr₀+20)"), (BOUNDARY_NMAX, "used on the surface")):
        ax.axvline(n, color=INK_3, lw=1.0, ls=(0, (3, 3)), zorder=2)
        # Top of the axis, not the bottom: the field-point traces bottom out
        # there and a low annotation lands on their labels.
        ax.annotate(f"{note}\nnmax = {n}", (n, 1e-4),
                    xytext=(-5 if n > 100 else 5, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_2,
                    ha="right" if n > 100 else "left", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_ticks(ax, "x", [4, 10, 25, 50, 100, 200], fmt=lambda v: f"{v:g}")
    _log_ticks(ax, "y", [1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e2],
               fmt=lambda v: f"{v:g}")
    ax.set_ylim(1e-13, 3e2)
    ax.set_xlabel("truncation order nmax", color=INK_2, fontsize=8.5)
    ax.set_ylabel(f"error against nmax = {top}, %", color=INK_2, fontsize=8.5)
    ax.set_title(f"Self-convergence at {frequency / 1000:g} kHz "
                 f"(ka = {2 * np.pi * frequency / 340 * 0.05:.1f})",
                 color=INK, fontsize=10, loc="left", pad=8)

    _style(bx)
    hz = np.geomspace(250.0, 16000.0, 24)
    ceiling = representable_ceiling(hz)
    auto = np.ceil(2 * np.pi * hz / 340.0 * (RADIUS_MM / 1e3)) + 20
    bx.fill_between(hz, auto, ceiling, color=RAMP[0], alpha=0.45, zorder=2,
                    lw=0)
    bx.plot(hz, ceiling, color=HUES[0], lw=2.0, zorder=3)
    bx.plot(hz, auto, color=RAMP[3], lw=2.0, zorder=3)
    bx.annotate("overflows above this", (hz[-1], ceiling[-1]), xytext=(-4, 7),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=HUES[0])
    bx.annotate("auto-selected", (hz[-1], auto[-1]), xytext=(-4, -13),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=RAMP[3])
    bx.annotate("usable", (1400, 70), fontsize=8, color=INK_2)
    bx.set_xscale("log")
    _log_ticks(bx, "x", [250, 500, 1000, 2000, 4000, 8000, 16000])
    bx.set_xlabel("frequency, Hz", color=INK_2, fontsize=8.5)
    bx.set_ylabel("truncation order nmax", color=INK_2, fontsize=8.5)
    bx.set_title("The usable window", color=INK, fontsize=10, loc="left",
                 pad=8)

    fig.suptitle("How good is the reference? The closed form against itself",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(0.02, 0.905,
             "Field points are exact by nmax 45. The surface is not: it "
             "converges algebraically and stops on a Gibbs floor at the "
             "velocity jump.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plot", action="store_true",
                    help="print the tables only; needs no matplotlib")
    args = ap.parse_args(argv)

    models = load_study(HERE)
    rows = [m for m in (measure(HERE, n, models[n]) for n in VARIANTS)
            if m is not None]
    if not rows:
        sys.exit(f"No solved variants under {HERE / 'runs'} — "
                 f"run `bem {HERE} solve` first.")

    print_table(rows)
    print_secondary(rows)
    print_checks(rows)

    if not args.no_plot:
        print()
        plot_frequency(rows, HERE / "error_vs_frequency.png")
        plot_cost(rows, HERE / "error_vs_elements.png")
        plot_truncation(HERE / "analytic_truncation.png")


if __name__ == "__main__":
    main()
