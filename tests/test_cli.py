import argparse

import pytest

from bemkit.cli import main


def _parse(argv):
    """Parse argv through a parser built like the real CLI."""
    from bemkit.cli import _pipeline, _plot, _viz3d, _web
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?")
    sub = parser.add_subparsers(dest="command")
    for mod in (_pipeline, _plot, _viz3d, _web):
        mod.register(sub)
    return parser.parse_args(argv)


def _variant_of(argv):
    """Parse argv through a parser built like the real CLI, return .variant."""
    return _parse(argv).variant


@pytest.mark.parametrize("command", ["geometry", "results"])
def test_symmetry_mirroring_is_on_by_default_in_both_viewers(command):
    """A [SYMMETRY] model shows a partial surface next to a full-space eval
    grid unless the ghosts are drawn, so mirroring is opt-out, not opt-in."""
    assert _parse([command]).no_mirror is False
    assert _parse([command, "--no-mirror"]).no_mirror is True


@pytest.mark.parametrize("command", ["geometry", "results"])
def test_mesh_edges_are_drawn_by_default_in_both_viewers(command):
    """Edges mark which elements were actually simulated — the only cue left
    once ghosts are drawn at full opacity, so they are opt-out."""
    assert _parse([command]).no_edges is False
    assert _parse([command, "--no-edges"]).no_edges is True


class TestRunFlags:
    """`run` detaches by default; --foreground is both the escape hatch and
    the recursion guard the background supervisor re-invokes itself with."""

    def test_foreground_is_off_by_default(self):
        assert _parse(["run"]).foreground is False
        assert _parse(["run", "--foreground"]).foreground is True

    def test_restart_is_off_by_default(self):
        assert _parse(["run"]).restart is False
        assert _parse(["run", "--restart"]).restart is True

    def test_restart_is_independent_of_force(self):
        """--restart kills and resumes; --force recomputes. Orthogonal, so
        --force must not be implied (it would rmtree be.out/ unasked)."""
        assert _parse(["run", "--restart"]).force is False
        args = _parse(["run", "--restart", "--force"])
        assert (args.restart, args.force) == (True, True)


class TestPackFlags:
    """Packing deletes be.out/ by default now — it is gated on a verified
    round-trip and reversible via unpack_variant."""

    def test_keep_original_is_off_by_default(self):
        assert _parse(["pack"]).keep_original is False
        assert _parse(["pack", "--keep-original"]).keep_original is True

    def test_delete_original_is_gone(self):
        """It is the default, so silently accepting it would be fine — but it
        also read as 'and this is the unusual, dangerous one', which is now
        backwards. Fail loudly instead."""
        with pytest.raises(SystemExit):
            _parse(["pack", "--delete-original"])


def test_geometry_still_accepts_the_superseded_show_edges_flag():
    """--show-edges is what edges used to need; it is the default now, so the
    flag stays valid and does nothing rather than breaking old invocations."""
    assert _parse(["geometry", "--show-edges"]).no_edges is False


@pytest.mark.parametrize("command", ["run", "export", "pack", "all",
                                     "plot", "geometry", "results", "export-web"])
def test_variant_flag_accumulates_across_repeats_and_spaces(command):
    # The bug: `--variant a --variant b` used to keep only ['b'] (last wins).
    base = [command] if command != "export-web" else [command, "--out", "d"]
    assert _variant_of(base + ["--variant", "a", "--variant", "b"]) == ["a", "b"]
    assert _variant_of(base + ["--variant", "a", "b"]) == ["a", "b"]
    assert _variant_of(base + ["--variant", "a"]) == ["a"]
    assert _variant_of(base) is None


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["x", "init", "--help"],
    ["x", "mesh", "--help"],
    ["x", "export", "--help"],
    ["x", "run", "--help"],
    ["x", "pack", "--help"],
    ["x", "ps", "--help"],
    ["x", "plot", "--help"],
    ["x", "geometry", "--help"],
    ["x", "results", "--help"],
])
def test_help_exits_zero(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_status_on_model_without_vars_toml(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


def test_python_dash_m_invocation():
    """`python -m bemkit.cli` works without installing the bem entry point."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "bemkit.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "bem" in r.stdout


def test_explicit_model_dir_with_no_command_still_parses(tmp_path):
    """Regression guard: model_dir stays a required single-token positional
    internally (see main()) specifically so that `bem MODEL_DIR` with no
    COMMAND keeps parsing correctly. Exit code must come from the "no
    config.toml" check, not from argparse failing to parse the arguments
    (argparse usage errors exit with code 2)."""
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path)])
    assert exc.value.code != 2


def test_model_dir_defaults_to_cwd_no_args(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.toml").write_text("[bem]\nsr = 32000\nnbins = 65\n")
    monkeypatch.chdir(tmp_path)
    main([])
    out = capsys.readouterr().out
    assert str(tmp_path.resolve()) in out


def test_model_dir_defaults_to_cwd_with_command(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text("[bem]\nsr = 32000\nnbins = 65\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["mesh", "--help"])
    assert exc.value.code == 0


def test_model_dir_named_like_subcommand_needs_path_prefix(tmp_path, monkeypatch, capsys):
    """A model directory literally named "mesh" collides with the "mesh"
    subcommand name: bare `bem mesh` means the subcommand (targeting cwd,
    which has no config.toml here), not the directory. `bem ./mesh`
    disambiguates and correctly targets the directory."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / "config.toml").write_text("[bem]\nsr = 32000\nnbins = 65\n")
    monkeypatch.chdir(tmp_path)

    # bare "mesh" is the subcommand, targeting cwd (tmp_path) -> no config.toml there
    with pytest.raises(SystemExit):
        main(["mesh"])

    # "./mesh" is unambiguously a path -> targets mesh_dir, which does have config.toml
    main(["./mesh"])
    out = capsys.readouterr().out
    assert str(mesh_dir.resolve()) in out
