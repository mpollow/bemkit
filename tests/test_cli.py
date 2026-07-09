import pytest

from pynumcalc.cli import main


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
    """`python -m pynumcalc.cli` works without installing the bem entry point."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pynumcalc.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "bem" in r.stdout
