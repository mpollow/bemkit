"""Convergence plots: how well NumCalc actually solved, bin by bin.

`bem STUDY status` answers whether a bin is *present*. These plots answer
whether it is *trustworthy* — how many CGS iterations it took, what residual it
reached, and what that cost in wall-clock seconds.

The data is what NumCalc prints once per frequency step; there is no
per-iteration residual trace in its logs, so the x-axis here is frequency and
never iteration number. :func:`numcalckit.status.convergence` does the reading;
this module is only the drawing.

The layout is an **envelope**, not one colour per variant: a study routinely has
more variants than any categorical palette has slots (23 here), so individual
variants are thin grey traces and the study as a whole is a min-max band with a
median through it. Only the outliers get a colour and a name — which is the
whole point, since an outlier is the bin worth looking at.
"""

import numpy as np

from numcalckit.status import CGS_TOLERANCE, NITER_MAX_DEFAULT

#: Type scale. matplotlib's defaults are sized for a small figure; on a
#: 13-inch canvas they swamp the data, which is the one thing the reader is
#: here for. Same values as the study scripts, so the figures read as one set.
RC = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
      "figure.titlesize": 10, "axes.labelcolor": "#3a3a37",
      "text.color": "#3a3a37", "xtick.color": "#6a6a67",
      "ytick.color": "#6a6a67"}

INK = "#3a3a37"          #: primary text/median ink
MUTED = "#6a6a67"        #: secondary text
TRACE = "#b9b9b5"        #: one variant's own line
BAND = "#e8e8e5"         #: min-max envelope fill
GRID = "#e4e4e1"
SPINE = "#c9c9c5"

#: Categorical slots 1-3 of the reference palette — the three that clear the
#: all-pairs CVD floors in both modes. Used only for the three time components,
#: which are the only place this figure encodes identity by colour.
TIME_COLOURS = {"t_assemble": "#2a78d6", "t_solve": "#eb6834",
                "t_post": "#1baf7a"}
TIME_LABELS = {"t_assemble": "assembling", "t_solve": "solving",
               "t_post": "post-processing"}

#: Status-critical. Reserved for non-convergence and never used as a series
#: colour, and always paired with a legend entry and a text label — a reader
#: who cannot see the red must still be told.
CRITICAL = "#d03b3b"


def _grid(records_by_variant, attr):
    """``(bins, frequencies, values)`` — values is (n_variants, n_bins), NaN-filled.

    Keyed on bin index rather than on the frequency read out of the log: the
    bin is the canonical identity of a step, and two logs printing the same
    frequency to different precision must not become two columns.
    """
    bins = sorted({r.bin for recs in records_by_variant.values() for r in recs})
    index = {b: i for i, b in enumerate(bins)}
    freqs = np.full(len(bins), np.nan)
    values = np.full((len(records_by_variant), len(bins)), np.nan)
    for row, recs in enumerate(records_by_variant.values()):
        for r in recs:
            col = index[r.bin]
            if r.frequency is not None:
                freqs[col] = r.frequency
            v = getattr(r, attr)
            if v is not None:
                values[row, col] = v
    return np.asarray(bins), freqs, values


def _envelope(ax, x, values, *, single_name=None):
    """Min-max band + per-variant traces + median.

    With one variant there is no envelope to draw: the band is degenerate and
    the median *is* that variant's own trace, so labelling either "over
    variants" would claim a spread that is not there. It gets one line under
    its own name instead.
    """
    if single_name is not None:
        ax.plot(x, values[0], lw=1.8, color=INK, zorder=4, label=single_name)
        return
    with np.errstate(invalid="ignore"):
        lo = np.nanmin(values, axis=0)
        hi = np.nanmax(values, axis=0)
        mid = np.nanmedian(values, axis=0)
    good = np.isfinite(lo)
    if good.any():
        ax.fill_between(x[good], lo[good], hi[good], color=BAND, zorder=1,
                        label="min–max over variants")
    for row in values:
        ax.plot(x, row, lw=0.5, color=TRACE, alpha=0.55, zorder=2)
    ax.plot(x, mid, lw=2.0, color=INK, zorder=4, label="median over variants")


def _label_extreme(ax, names, x, values, *, fmt, largest=True):
    """Name the one variant that owns the extreme, at the bin where it does.

    One label, not a legend of 23: the reader needs to know *which* run to go
    look at, and every other trace is context for that one.
    """
    if not np.isfinite(values).any():
        return None
    flat = np.where(np.isnan(values), -np.inf if largest else np.inf, values)
    row, col = np.unravel_index(
        (np.argmax if largest else np.argmin)(flat), flat.shape)
    ax.plot(x[col], values[row, col], "o", ms=5, mfc="none", mec=INK, mew=1.2,
            zorder=6)
    # Flip the label inward once the point is past the middle: the extreme is
    # very often the top bin, where a right-hand label runs off the canvas.
    # The midpoint is geometric because the axis is log-frequency.
    late = x[col] >= np.sqrt(np.nanmin(x) * np.nanmax(x))
    ax.annotate(f"{names[row]}  ({fmt(values[row, col])})",
                (x[col], values[row, col]), textcoords="offset points",
                xytext=(-7 if late else 7, 4), fontsize=7.5, color=INK,
                ha="right" if late else "left", va="center", zorder=6)
    return names[row]


def _mark_warnings(ax, records_by_variant, y_for):
    """Ring and name every bin NumCalc warned about. Returns the variant names."""
    flagged = []
    for name, recs in records_by_variant.items():
        for r in recs:
            if not r.warnings or r.frequency is None:
                continue
            y = y_for(r)
            if y is None or not np.isfinite(y):
                continue
            ax.plot(r.frequency, y, "o", ms=7, mfc="none", mec=CRITICAL,
                    mew=1.6, zorder=7,
                    label="did not converge" if not flagged else None)
            ax.annotate(name, (r.frequency, y), textcoords="offset points",
                        xytext=(7, -8), fontsize=7.5, color=CRITICAL, zorder=7)
            flagged.append(name)
    return flagged


def _style(ax, *, log_y=False):
    ax.set_xscale("log")
    ax.grid(True, axis="x", which="both", lw=0.5, color=GRID, zorder=0)
    # A log y-axis has ~8 minor lines per decade; drawn at grid weight they
    # out-ink the data they are meant to sit behind.
    ax.grid(True, axis="y", which="major" if log_y else "both", lw=0.5,
            color=GRID, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def plot_convergence(records_by_variant, *, title=None,
                     niter_max=NITER_MAX_DEFAULT):
    """Three stacked panels over one shared frequency axis.

    Args:
        records_by_variant: ``{variant: [BinRecord, ...]}`` from
            :func:`numcalckit.status.convergence`. Variants with no records are
            dropped — a run whose logs were deleted has nothing to say here,
            and an empty row would read as a solve that took zero iterations.
        title: figure suptitle.
        niter_max: the iteration ceiling in force (``-nitermax``, or NumCalc's
            own default). Drawn only when the data comes close enough to it to
            be worth seeing.

    Returns:
        The matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    data = {k: v for k, v in records_by_variant.items() if v}
    if not data:
        raise ValueError(
            "no convergence records to plot — every variant's NC*.out logs are "
            "missing or empty. Logs are not checked in, and `bem solve --force` "
            "deletes them.")

    names = list(data)
    only = names[0] if len(names) == 1 else None
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)

    bins, freqs, iters = _grid(data, "iterations")
    _, _, errors = _grid(data, "rel_error")
    x = freqs

    # --- 1. iterations ----------------------------------------------------
    ax = axes[0]
    _envelope(ax, x, iters, single_name=only)
    worst_iter = _label_extreme(ax, names, x, iters, fmt=lambda v: f"{v:.0f}")
    _mark_warnings(ax, data, lambda r: r.iterations)
    ax.set_ylabel("CGS iterations")
    # The ceiling is 1500 by default and the data is typically under 120;
    # drawing it to scale would flatten every curve into the axis. Show it
    # only once it is close enough to matter, and say the headroom otherwise.
    peak = np.nanmax(iters) if np.isfinite(iters).any() else 0.0
    if niter_max and peak > 0.5 * niter_max:
        ax.axhline(niter_max, lw=1.0, ls="--", color=CRITICAL, zorder=3,
                   label=f"ceiling ({niter_max})")
    elif niter_max:
        ax.text(0.995, 0.04, f"ceiling {niter_max:,} — peak is {peak:.0f}",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    ax.set_title("How hard each bin was to solve", loc="left")

    # --- 2. final relative error ------------------------------------------
    ax = axes[1]
    ax.set_yscale("log")
    _envelope(ax, x, errors, single_name=only)
    _label_extreme(ax, names, x, errors, fmt=lambda v: f"{v:.1e}")
    _mark_warnings(ax, data, lambda r: r.rel_error)
    ax.axhline(CGS_TOLERANCE, lw=1.2, ls="--", color=CRITICAL, zorder=5,
               label=f"tolerance ({CGS_TOLERANCE:.0e})")
    if np.isfinite(errors).any():
        ax.set_ylim(top=max(np.nanmax(errors), CGS_TOLERANCE) * 3.0)
    ax.set_ylabel("final relative residual")
    ax.set_title("Where the iteration stopped — above the dashed line "
                 "means it did not converge", loc="left")

    # --- 3. cost -----------------------------------------------------------
    ax = axes[2]
    _, _, totals = _grid(data, "t_total")
    with np.errstate(invalid="ignore"):
        if only is not None:
            ax.plot(x, totals[0], lw=1.2, color=TRACE, zorder=1, label="total")
        elif np.isfinite(totals).any():
            ax.fill_between(x, np.nanmin(totals, axis=0),
                            np.nanmax(totals, axis=0), color=BAND, zorder=1,
                            label="min–max of total, over variants")
    for attr, colour in TIME_COLOURS.items():
        _, _, t = _grid(data, attr)
        with np.errstate(invalid="ignore"):
            ax.plot(x, t[0] if only else np.nanmedian(t, axis=0), lw=2.0,
                    color=colour, zorder=4,
                    label=TIME_LABELS[attr] if only
                    else f"median {TIME_LABELS[attr]}")
    ax.set_ylabel("seconds per bin")
    ax.set_xlabel("frequency [Hz]")
    ax.set_title("What it cost", loc="left")

    # Legend corners are picked per panel against where the data actually
    # sits: iterations and residual both rise to the right and carry a named
    # outlier near the top, so their legends go bottom-left; the cost panel is
    # empty top-left until the post-processing curve takes off.
    for ax, loc, log_y in zip(axes, ("lower left", "lower left", "upper left"),
                              (False, True, False)):
        _style(ax, log_y=log_y)
        ax.legend(frameon=False, loc=loc, fontsize=7.5)
    axes[0].margins(y=0.16)     # headroom for the outlier label

    warned = sorted(_warned_names(data))
    if title:
        fig.suptitle(title, fontsize=10)
    note = f"{len(data)} variant(s), {len(bins)} bins."
    if only is None:
        note += (" Grey traces are individual variants; "
                 "only outliers are named.")
    if warned:
        note += f"  Non-converged: {', '.join(warned)}."
    else:
        note += "  Every bin converged."
    if worst_iter and only is None:
        note += f"  Hardest run: {worst_iter}."
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965 if title else 1.0))
    return fig


def _warned_names(records_by_variant):
    return {name for name, recs in records_by_variant.items()
            if any(r.warnings for r in recs)}
