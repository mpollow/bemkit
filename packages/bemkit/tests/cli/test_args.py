"""Argument-parsing contracts of the `bem` CLI.

Rescued from the deleted ``tests/test_cli.py``, which tested the config.toml
stack's parser. The flags these pin — detach-by-default, pack-deletes-by-
default, the accumulating ``--variant`` — all survived the port onto the
model.py spine, so their defaults still have to be defended. What went with
the old parser is only the commands that no longer exist (``init``, ``mesh``,
``export``, ``all``).
"""

import argparse
import subprocess
import sys

import pytest

from bemkit.cli import main


def _parse(argv):
    """Parse argv through a parser built the way `bem` builds its own, but
    without the study_dir positional — so a bare ``["solve"]`` is enough."""
    from bemkit.cli import _check, _commands
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", nargs="?")
    sub = parser.add_subparsers(dest="command")
    _check.register(sub)
    _commands.register(sub)
    return parser.parse_args(argv)


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["x", "status", "--help"],
    ["x", "solve", "--help"],
    ["x", "estimate-ram", "--help"],
    ["x", "pack", "--help"],
    ["x", "ps", "--help"],
    ["x", "check", "--help"],
])
def test_help_exits_zero(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


class TestSolveFlags:
    """`solve` detaches by default; --foreground is both the escape hatch and
    the recursion guard the background supervisor re-invokes itself with."""

    def test_foreground_is_off_by_default(self):
        assert _parse(["solve"]).foreground is False
        assert _parse(["solve", "--foreground"]).foreground is True

    def test_restart_is_off_by_default(self):
        assert _parse(["solve"]).restart is False
        assert _parse(["solve", "--restart"]).restart is True

    def test_restart_is_independent_of_force(self):
        """--restart kills and resumes; --force recomputes. Orthogonal, so
        --force must not be implied (it would rmtree be.out/ unasked)."""
        assert _parse(["solve", "--restart"]).force is False
        args = _parse(["solve", "--restart", "--force"])
        assert (args.restart, args.force) == (True, True)

    def test_jobs_defaults_to_one(self):
        """BEM peak memory, not CPU, is the binding constraint."""
        assert _parse(["solve"]).jobs == 1
        assert _parse(["solve", "-j", "4"]).jobs == 4

    def test_band_selectors_default_to_none_and_are_exclusive(self):
        args = _parse(["solve"])
        assert (args.freq, args.bins, args.preview) == (None, None, None)
        with pytest.raises(SystemExit):
            _parse(["solve", "--freq", "1000", "--preview", "4"])


class TestPackFlags:
    """Packing deletes be.out/ by default — it is gated on a verified
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


@pytest.mark.parametrize("command", ["solve", "pack", "estimate-ram"])
def test_variant_flag_accumulates_across_repeats_and_spaces(command):
    """The bug: `--variant a --variant b` used to keep only ['b'] (argparse's
    last-wins on nargs='+')."""
    assert _parse([command, "--variant", "a", "--variant", "b"]).variant == ["a", "b"]
    assert _parse([command, "--variant", "a", "b"]).variant == ["a", "b"]
    assert _parse([command, "--variant", "a"]).variant == ["a"]
    assert _parse([command]).variant is None


def test_python_dash_m_invocation():
    """`python -m bemkit.cli` is the no-install entry point, and is what the
    detached supervisor re-execs — so it has to keep working."""
    r = subprocess.run([sys.executable, "-m", "bemkit.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "STUDY_DIR" in r.stdout


def test_explicit_study_dir_with_no_command_shows_status(tmp_path, capsys):
    """`bem PATH` with no subcommand is the status command."""
    (tmp_path / "model.py").write_text(
        "from bemkit.model import Mesh, Medium, Model, fft_grid\n"
        "import numpy as np\n"
        "def build():\n"
        "    mesh = Mesh(np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),\n"
        "                np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]]))\n"
        "    return Model(mesh, fft_grid(8000, 3), Medium())\n")
    main([str(tmp_path)])
    assert "export" in capsys.readouterr().out
