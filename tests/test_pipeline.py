"""RunStage glue: be.npz clearing, stale-log removal, solver-warning report.

Uses a fake NumCalc that mimics the real binary's semantics: unranged runs
wipe be.out/ and log to NC.out, ranged runs preserve be.out/ and log to
NC{istart}-{iend}.out.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from pynumcalc.config import ModelConfig
from pynumcalc.pipeline import RunStage

GOLDEN_MODEL = Path(__file__).parent / "golden" / "model"

_FAKE_NUMCALC = """\
#!/bin/sh
echo "$@" >> args.txt
istart=1; iend=4; ranged=0
while [ $# -gt 0 ]; do
  case $1 in
    -istart) istart=$2; ranged=1; shift 2;;
    -iend) iend=$2; shift 2;;
    *) shift;;
  esac
done
if [ $ranged -eq 1 ]; then log="NC$istart-$iend.out"; else log="NC.out"; rm -rf be.out; fi
echo "run" > "$log"
if [ -f warnplease ]; then
  echo "Step $istart, Frequency = 100 Hz" >> "$log"
  echo "Warning: Maximum number of iterations is reached!" >> "$log"
fi
i=$istart
while [ $i -le $iend ]; do mkdir -p be.out/be.$i; i=$((i+1)); done
"""


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    """Golden-model copy with a numcalc/tetra/ dir and a fake NumCalc on PATH."""
    d = tmp_path / "model"
    shutil.copytree(GOLDEN_MODEL, d)
    (d / "numcalc" / "tetra").mkdir(parents=True)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "NumCalc"
    exe.write_text(_FAKE_NUMCALC)
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return d


def _make_npz(nc_dir, bins, nf=4):
    arr = np.full((3, nf), np.nan, dtype=np.complex64)
    arr[:, [b - 1 for b in bins]] = 1.0
    np.savez(nc_dir / "be.npz", pEvalGrid=arr,
             bin_indices=np.array(bins, dtype=np.int32))


def _stage_run(model_dir, **kw):
    return RunStage().run(ModelConfig.from_dir(model_dir), **kw)


class TestRunStage:
    def test_full_run_clears_stale_npz_and_logs(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2])
        (nc / "NC3-4.out").write_text(
            "Step 3, Frequency = 750 Hz\n"
            "Warning: Maximum number of iterations is reached!\n")

        completed, failed = _stage_run(model_dir)

        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert not (nc / "be.npz").exists()
        assert not (nc / "NC3-4.out").exists()  # stale log removed
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]

    def test_missing_only_restores_npz_and_keeps_logs(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2])
        (nc / "NC.out").write_text("Step 1, Frequency = 100 Hz\nfine\n")

        completed, failed = _stage_run(model_dir, missing_only=True)

        assert failed == []
        assert not (nc / "be.npz").exists()
        assert (nc / "NC.out").exists()  # resume keeps earlier logs
        # only the gap was computed; bins 1-2 came back from the npz
        assert (nc / "args.txt").read_text().split() == \
            ["-istart", "3", "-iend", "4"]
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]

    def test_missing_only_complete_npz_untouched(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2, 3, 4])

        completed, failed = _stage_run(model_dir, missing_only=True)

        assert (completed, failed) == ([], [])
        assert (nc / "be.npz").exists()
        assert not (nc / "args.txt").exists()

    def test_full_run_over_complete_refuses_without_force(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2, 3, 4])  # complete
        with pytest.raises(RuntimeError, match="already complete"):
            _stage_run(model_dir)
        assert (nc / "be.npz").exists()      # results untouched
        assert not (nc / "args.txt").exists()  # NumCalc never invoked

    def test_force_allows_full_rerun_over_complete(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2, 3, 4])
        completed, failed = _stage_run(model_dir, force=True)
        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert not (nc / "be.npz").exists()          # moved aside
        assert (nc / "be.npz.bak").exists()          # backup kept
        assert (nc / "args.txt").exists()            # NumCalc ran

    def test_reports_solver_warnings_after_run(self, model_dir, capsys):
        nc = model_dir / "numcalc" / "tetra"
        (nc / "warnplease").touch()

        _stage_run(model_dir)

        out = capsys.readouterr().out
        assert "1 solver warning(s)" in out and "bins 1" in out

    def test_no_warning_report_for_clean_run(self, model_dir, capsys):
        _stage_run(model_dir)
        assert "solver warning" not in capsys.readouterr().out


class TestPackStage:
    def test_empty_be_out_skipped_not_aborting(self, tmp_path):
        # A variant with an empty be.out/ (exported, never run) must be skipped,
        # not abort packing of the other variants in the same model.
        import meshio
        from pynumcalc.config import ModelConfig
        from pynumcalc.pipeline import PackStage

        model = tmp_path / "m"
        (model / "mesh").mkdir(parents=True)
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        for stem in ("good", "empty"):
            meshio.write_points_cells(model / "mesh" / f"{stem}.stl", verts,
                                      [("triangle", tris)])
        (model / "vars.toml").write_text("sr = 8000\nnbins = 5\n")

        # good/: two real bins with scalar boundary + eval quantities
        for bi in (1, 2):
            d = model / "numcalc" / "good" / "be.out" / f"be.{bi}"
            d.mkdir(parents=True)
            for q in ("pBoundary", "pEvalGrid"):
                rows = "".join(f"{j} {bi*10+j}.0 {bi*100+j}.0\n" for j in range(3))
                (d / q).write_text("h1\nh2\nh3\n" + rows)
        # empty/: be.out dir with no bin results
        (model / "numcalc" / "empty" / "be.out").mkdir(parents=True)

        packed = PackStage().run(ModelConfig.from_dir(model))  # must not raise
        assert "good" in packed and "empty" not in packed
        assert (model / "numcalc" / "good" / "be.npz").exists()
        assert not (model / "numcalc" / "empty" / "be.npz").exists()
