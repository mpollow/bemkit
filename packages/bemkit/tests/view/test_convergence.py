"""The convergence figure.

Headless throughout: ``plot_convergence`` takes plain ``BinRecord`` lists, so
none of this needs a study directory, a solver, or a display.
"""

import matplotlib
matplotlib.use("Agg")

import pytest

from pynumcalc.status import BinRecord
from bemkit.view.convergence import CRITICAL, plot_convergence


def _records(n=6, *, iters=50, warn_at=None):
    """One variant's worth of records over a log-spaced band."""
    return [BinRecord(bin=i + 1, frequency=250.0 * 2 ** i,
                      iterations=iters + i, rel_error=1e-10,
                      t_assemble=40.0, t_solve=20.0, t_post=15.0,
                      t_total=75.0,
                      warnings=(("Warning: Maximum number of iterations "
                                 "is reached!",) if i + 1 == warn_at else ()))
            for i in range(n)]


def test_three_panels_over_one_frequency_axis():
    fig = plot_convergence({"a": _records(), "b": _records(iters=80)})
    axes = fig.axes
    assert len(axes) == 3
    assert [ax.get_xscale() for ax in axes] == ["log"] * 3
    # Only the residual panel is log-y; iterations and seconds are linear.
    assert [ax.get_yscale() for ax in axes] == ["linear", "log", "linear"]
    assert axes[2].get_xlabel() == "frequency [Hz]"


def test_every_panel_carries_a_legend():
    """With 23 grey traces on an axis, the band and the median are only
    readable if they are named."""
    fig = plot_convergence({"a": _records(), "b": _records(iters=80)})
    for ax in fig.axes:
        assert ax.get_legend() is not None


def test_variants_with_no_records_are_dropped_not_drawn_as_zero():
    """A variant whose logs were deleted has nothing to say; an empty row
    would read as a solve that took no iterations."""
    fig = plot_convergence({"a": _records(), "empty": []})
    assert "1 variant(s)" in _footnote(fig)


def test_a_study_with_nothing_to_plot_is_refused_by_name():
    with pytest.raises(ValueError, match="NC..out logs are missing"):
        plot_convergence({"a": [], "b": []})


class TestASingleVariantClaimsNoSpread:
    """The band is degenerate and the median *is* that variant's own trace, so
    calling either one "over variants" would assert a spread that is not
    there."""

    def test_the_line_carries_the_variant_name(self):
        fig = plot_convergence({"solo": _records()})
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert "solo" in labels
        assert not any("over variants" in t for t in labels)

    def test_the_footnote_drops_the_grey_trace_explanation(self):
        note = _footnote(plot_convergence({"solo": _records()}))
        assert "Grey traces" not in note
        assert "Hardest run" not in note      # trivially itself


def test_the_hardest_variant_is_named_on_the_figure():
    """One direct label, not a legend of every variant: the reader needs to
    know which run to go look at."""
    fig = plot_convergence({"easy": _records(iters=40),
                            "hard": _records(iters=90)})
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert any("hard" in t for t in labels)
    assert "Hardest run: hard" in _footnote(fig)


class TestNonConvergenceIsNeverColourAlone:
    """A reader who cannot distinguish the red must still be told."""

    def test_the_marker_is_status_critical(self):
        fig = plot_convergence({"a": _records(warn_at=3)})
        assert any(line.get_markeredgecolor() == CRITICAL
                   for line in fig.axes[0].lines)

    def test_the_bin_is_also_labelled_with_the_variant_name(self):
        fig = plot_convergence({"a": _records(warn_at=3)})
        assert any(t.get_text() == "a" for t in fig.axes[0].texts)

    def test_and_the_footnote_names_it(self):
        fig = plot_convergence({"a": _records(warn_at=3), "b": _records()})
        assert "Non-converged: a" in _footnote(fig)

    def test_a_clean_study_says_so_rather_than_staying_silent(self):
        assert "Every bin converged" in _footnote(plot_convergence({"a": _records()}))


class TestTheIterationCeiling:
    """1500 by default against data under 120: drawn to scale it would flatten
    every curve into the axis, so it is only drawn once it is close."""

    def test_far_below_the_ceiling_it_is_stated_in_text(self):
        fig = plot_convergence({"a": _records(iters=50)}, niter_max=1500)
        assert any("ceiling" in t.get_text() for t in fig.axes[0].texts)
        assert not _has_hline(fig.axes[0], 1500)

    def test_close_to_the_ceiling_it_is_drawn(self):
        fig = plot_convergence({"a": _records(iters=1400)}, niter_max=1500)
        assert _has_hline(fig.axes[0], 1500)


def _has_hline(ax, y):
    return any(line.get_ydata()[0] == y and len(set(line.get_ydata())) == 1
               for line in ax.lines if len(line.get_ydata()))


def _footnote(fig):
    return " ".join(t.get_text() for t in fig.texts)
