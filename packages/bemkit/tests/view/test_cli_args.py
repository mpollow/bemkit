"""The view layer's `bem` subcommand parsers.

Rescued from the deleted ``packages/bemkit/tests/test_cli.py``: these pin the
view verbs' flag defaults, which survived the config.toml deletion intact.

The registration seam these go through changed when ``bemviz`` was folded into
``bemkit.view``: it is a direct call from ``bemkit.cli.main`` now, not an
entry point. That is why the `--help` test below is unconditional — it used to
carry a skipif on the entry point being installed, which after the fold would
have quietly skipped instead of testing anything.
"""

import argparse

import pytest

from bemkit import view


def _parse(argv):
    """Parse argv through a parser built the way `bem` builds its own."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", nargs="?")
    sub = parser.add_subparsers(dest="command")
    view.register_cli(sub)
    return parser.parse_args(argv)


@pytest.mark.parametrize("argv", [["x", "geometry", "--help"],
                                  ["x", "results", "--help"],
                                  ["x", "export-web", "--help"]])
def test_help_exits_zero_through_the_real_cli(argv):
    """--help must work for every view subcommand, including export-web's,
    whose body still refuses: the flags are still the documented contract."""
    from bemkit.cli import main
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 0


@pytest.mark.parametrize("command", ["geometry", "results"])
def test_symmetry_mirroring_is_on_by_default_in_both_viewers(command):
    """A mirrored model shows a partial surface next to a full-space eval
    grid unless the ghosts are drawn, so mirroring is opt-out, not opt-in."""
    assert _parse([command]).no_mirror is False
    assert _parse([command, "--no-mirror"]).no_mirror is True


@pytest.mark.parametrize("command", ["geometry", "results"])
def test_mesh_edges_are_drawn_by_default_in_both_viewers(command):
    """Edges mark which elements were actually simulated — the only cue left
    once ghosts are drawn at full opacity, so they are opt-out."""
    assert _parse([command]).no_edges is False
    assert _parse([command, "--no-edges"]).no_edges is True


def test_geometry_still_accepts_the_superseded_show_edges_flag():
    """--show-edges is what edges used to need; it is the default now, so the
    flag stays valid and does nothing rather than breaking old invocations."""
    assert _parse(["geometry", "--show-edges"]).no_edges is False


@pytest.mark.parametrize("command", ["geometry", "results", "export-web"])
def test_variant_flag_accumulates_across_repeats_and_spaces(command):
    """bemkit's VariantAction, exercised through the view parsers.

    The bug it exists to prevent: `--variant a --variant b` used to keep only
    ['b'] (argparse's last-wins on nargs='+').
    """
    base = [command] if command != "export-web" else [command, "--out", "d"]
    assert _parse(base + ["--variant", "a", "--variant", "b"]).variant == ["a", "b"]
    assert _parse(base + ["--variant", "a", "b"]).variant == ["a", "b"]
    assert _parse(base + ["--variant", "a"]).variant == ["a"]
    assert _parse(base).variant is None


def test_results_defaults_to_raw_not_acc():
    """acc is undefined for a scattering model, so it cannot be the default
    the plain invocation uses — `bem STUDY results` must work on a plane-wave
    study without a flag."""
    assert _parse(["results"]).norm == "raw"
    assert _parse(["results", "--norm", "acc"]).norm == "acc"


class TestResultsRefusesBeforeItOpensAWindow:
    """Everything decidable from the arguments alone is decided up front, so
    the failure reaches the terminal instead of a blank plotter."""

    def test_a_missing_study_is_a_message_not_a_traceback(self, tmp_path):
        from bemkit.view.cli_viz3d import cmd_results
        with pytest.raises(SystemExit) as e:
            cmd_results(tmp_path / "nope", _parse(["results"]))
        assert "not found" in str(e.value)

    def test_hiding_everything_is_refused(self, tmp_path):
        from bemkit.view.cli_viz3d import cmd_results
        args = _parse(["results", "--no-surface", "--no-points"])
        with pytest.raises(SystemExit) as e:
            cmd_results(tmp_path, args)
        assert "Nothing to visualize" in str(e.value)

    def test_a_study_with_no_solved_bins_says_so(self, tmp_path):
        """A run directory holding only an export has nothing to colour."""
        from bemkit.view.cli_viz3d import cmd_results
        (tmp_path / "model.py").write_text(
            "from bemkit.model import Mesh, Model\n"
            "import numpy as np\n"
            "def build():\n"
            "    v = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])\n"
            "    t = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])\n"
            "    return Model(Mesh(v, t), [100.0])\n")
        with pytest.raises(SystemExit) as e:
            cmd_results(tmp_path, _parse(["results"]))
        assert "solve" in str(e.value)


class TestExportWebStillRefusesCleanly:
    """It was built on the config.toml stack and has not been ported. A clean
    refusal naming the gap is the contract; an ImportError or an
    AttributeError from deep inside is not.
    """

    def test_export_web_exits_with_the_reason(self, tmp_path):
        from bemkit.view.cli_web import cmd_export_web
        args = _parse(["export-web", "--out", str(tmp_path / "bundle")])
        with pytest.raises(SystemExit) as e:
            cmd_export_web(tmp_path, args)
        assert "not yet ported" in str(e.value)

    def test_export_web_the_library_function_raises_rather_than_exits(self,
                                                                     tmp_path):
        """Called from Python (not the CLI) it must raise, so a caller can
        catch it — sys.exit inside a library function is not catchable as an
        error."""
        from bemkit.view.web import export_web
        with pytest.raises(NotImplementedError, match="not yet ported"):
            export_web(tmp_path, tmp_path / "bundle")
