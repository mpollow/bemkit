import pytest

from bemkit.cli import main


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["x", "mesh", "--help"],
    ["x", "export", "--help"],
    ["x", "run", "--help"],
    ["x", "pack", "--help"],
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
    (tmp_path / "config.toml").write_text("sr = 32000\nnbins = 65\n")
    monkeypatch.chdir(tmp_path)
    main([])
    out = capsys.readouterr().out
    assert str(tmp_path.resolve()) in out


def test_model_dir_defaults_to_cwd_with_command(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text("sr = 32000\nnbins = 65\n")
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
    (mesh_dir / "config.toml").write_text("sr = 32000\nnbins = 65\n")
    monkeypatch.chdir(tmp_path)

    # bare "mesh" is the subcommand, targeting cwd (tmp_path) -> no config.toml there
    with pytest.raises(SystemExit):
        main(["mesh"])

    # "./mesh" is unambiguously a path -> targets mesh_dir, which does have config.toml
    main(["./mesh"])
    out = capsys.readouterr().out
    assert str(mesh_dir.resolve()) in out
