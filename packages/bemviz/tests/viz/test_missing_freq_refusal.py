"""`bem MODEL plot ... --freq/--freqs` must refuse a frequency whose bin was
never computed, rather than silently plotting an all-NaN curve.

Verified live: with bins {1,2,4,8} computed, `bem MODEL plot polar --freqs
4000` produced a legend entry, an invisible curve, and a raw
``RuntimeWarning: All-NaN slice encountered`` — the CLI dispatcher only
catches ``IncompleteBand`` (and friends), so this checks the CLI raises that,
not that the underlying data happens to be NaN.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bemkit.analysis.result import BEMResult, IncompleteBand


def _sparse_result(missing=(2,), nf=4):
    n = 3
    p = np.ones((n, nf), dtype=complex)
    for b in missing:
        p[:, b - 1] = np.nan
    return BEMResult(
        name="t",
        frequencies=np.array([1000.0, 2000.0, 3000.0, 4000.0])[:nf],
        eval_points=np.zeros((n, 3)),
        pressure_raw=p, pressure_acc=p,
        velocity_raw=p, velocity_acc=p,
        rho0=1.2, speed_of_sound=340.0,
    )


def _args(**over):
    base = dict(freq=None, freqs=None, angles=None, ref=None, clim=None,
                contours=6.0, linear_freq=False, value_func="magnitude_db",
                filled=False)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _fake_load_result(monkeypatch, request):
    """Every test in this module supplies its own result via a marker-free
    hook: tests call this fixture's setter before invoking _run_plot."""
    holder = {}

    def _set(result):
        holder["result"] = result

    def _fake(model_dir, set_name):
        return holder["result"], "v"

    monkeypatch.setattr("bemkit.analysis.result.load_result", _fake)
    return _set


@pytest.mark.parametrize("plot_type", ["plane", "sphere", "contour"])
def test_refuses_a_freq_argument_that_maps_to_a_missing_bin(
        _fake_load_result, tmp_path, plot_type):
    # frequencies = [1000, 2000, 3000, 4000]; bin 2 (2000 Hz) is missing.
    _fake_load_result(_sparse_result(missing=(2,)))

    from bemviz.cli_plot import _run_plot
    with pytest.raises(IncompleteBand) as exc:
        _run_plot(plot_type, tmp_path, None, _args(freq=2000.0))
    assert "2000" in str(exc.value)


def test_names_the_nearest_available_frequency(_fake_load_result, tmp_path):
    _fake_load_result(_sparse_result(missing=(2,)))

    from bemviz.cli_plot import _run_plot
    with pytest.raises(IncompleteBand) as exc:
        _run_plot("sphere", tmp_path, None, _args(freq=2000.0))
    # 1000 Hz (bin 1) is the nearest computed neighbour of the missing 2000 Hz.
    assert "1000" in str(exc.value)


def test_a_computed_freq_passes_the_check(_fake_load_result, tmp_path):
    """No IncompleteBand for a frequency whose bin is actually present —
    whatever happens next (here: not grid-structured) is a different error."""
    _fake_load_result(_sparse_result(missing=(2,)))

    from bemviz.cli_plot import _run_plot
    with pytest.raises(SystemExit) as exc:
        _run_plot("sphere", tmp_path, None, _args(freq=1000.0))
    assert "grid" in str(exc.value)


def test_refuses_a_polar_freqs_entry_that_maps_to_a_missing_bin(
        _fake_load_result, tmp_path):
    _fake_load_result(_sparse_result(missing=(2,)))

    from bemviz.cli_plot import _run_plot
    with pytest.raises(IncompleteBand) as exc:
        _run_plot("polar", tmp_path, None, _args(freqs="2000"))
    assert "2000" in str(exc.value)


def test_plot_dispatcher_reports_skipped_not_a_traceback(
        _fake_load_result, tmp_path, monkeypatch, capsys):
    """The per-set loop (_plot_sets) already catches IncompleteBand and
    prints 'Skipped: ...' — confirm that still covers this new raise site."""
    from bemviz import cli_plot as plot_mod

    monkeypatch.setattr("bemkit.sim.discovery.discover_variants",
                        lambda *a, **k: ["v"])
    _fake_load_result(_sparse_result(missing=(2,)))
    args = _args(freq=2000.0)
    plot_mod._plot_sets(tmp_path, "sphere", None, args)

    out = capsys.readouterr().out
    assert "Skipped:" in out
    assert "2000" in out
