"""bemviz's `bem` subcommand parsers.

Rescued from the deleted ``packages/bemkit/tests/test_cli.py``: these pin
bemviz's own flag defaults, which survived the config.toml deletion intact
even though the commands behind two of them do not run yet. They live here
rather than in bemkit's suite because they are assertions about bemviz's
parsers — bemkit only provides the entry-point seam they plug into.
"""

import argparse

import pytest

import bemviz


def _parse(argv):
    """Parse argv through a parser built the way `bem` builds its own."""
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", nargs="?")
    sub = parser.add_subparsers(dest="command")
    bemviz.register_cli(sub)
    return parser.parse_args(argv)


def _bemviz_cli_installed():
    """True when bemviz's entry point is registered, not merely importable —
    `bem` discovers subcommands through the entry-point group."""
    from importlib.metadata import entry_points
    return any(e.value.startswith("bemviz")
               for e in entry_points(group="bemkit.cli"))


@pytest.mark.skipif(not _bemviz_cli_installed(),
                    reason="bemviz not installed (entry point absent)")
@pytest.mark.parametrize("argv", [["x", "geometry", "--help"],
                                  ["x", "results", "--help"],
                                  ["x", "export-web", "--help"]])
def test_help_exits_zero_through_the_real_cli(argv):
    """--help must work for every bemviz subcommand, including the two whose
    bodies currently refuse: the flags are still the documented contract."""
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
    """bemkit's VariantAction, exercised through bemviz's parsers.

    The bug it exists to prevent: `--variant a --variant b` used to keep only
    ['b'] (argparse's last-wins on nargs='+').
    """
    base = [command] if command != "export-web" else [command, "--out", "d"]
    assert _parse(base + ["--variant", "a", "--variant", "b"]).variant == ["a", "b"]
    assert _parse(base + ["--variant", "a", "b"]).variant == ["a", "b"]
    assert _parse(base + ["--variant", "a"]).variant == ["a"]
    assert _parse(base).variant is None


class TestUnportedCommandsRefuseCleanly:
    """`results` and `export-web` exit with a message, never a traceback.

    They were built on the config.toml stack's SimulationData/BEMResult. A
    clean refusal naming the gap is the contract until they are ported; an
    ImportError or an AttributeError from deep inside is not.
    """

    def test_results_exits_with_the_reason(self, tmp_path, capsys):
        from bemviz.cli_viz3d import cmd_results
        with pytest.raises(SystemExit) as e:
            cmd_results(tmp_path, _parse(["results"]))
        message = str(e.value)
        assert "not yet ported" in message
        assert "geometry" in message          # names what does work

    def test_export_web_exits_with_the_reason(self, tmp_path):
        from bemviz.cli_web import cmd_export_web
        args = _parse(["export-web", "--out", str(tmp_path / "bundle")])
        with pytest.raises(SystemExit) as e:
            cmd_export_web(tmp_path, args)
        assert "not yet ported" in str(e.value)

    def test_export_web_the_library_function_raises_rather_than_exits(self,
                                                                     tmp_path):
        """Called from Python (not the CLI) it must raise, so a caller can
        catch it — sys.exit inside a library function is not catchable as an
        error."""
        from bemviz.web import export_web
        with pytest.raises(NotImplementedError, match="not yet ported"):
            export_web(tmp_path, tmp_path / "bundle")
