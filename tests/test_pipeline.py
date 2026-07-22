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

from bemkit.config import ModelConfig
from bemkit.pipeline import MeshStage, RunStage

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


class TestMeshStage:
    """geometry/ holds remesh sources (CAD + scans); mesh/ is always trusted."""

    @pytest.fixture()
    def mesh_model(self, tmp_path, monkeypatch):
        """Sparse model dir + fake mesh generators recording their input names."""
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.toml").write_text("[bem]\nsr = 8000\nnbins = 5\n")
        calls = []

        def fake_gen(input_file, output_file, **kw):
            calls.append(Path(input_file).name)
            Path(output_file).touch()
            return Path(output_file)

        import bemkit.meshing.generate as generate
        monkeypatch.setattr(generate, "mesh_from_step", fake_gen)
        monkeypatch.setattr(generate, "mesh_from_surface", fake_gen)
        return model, calls

    def test_ply_in_mesh_dir_is_trusted_not_remeshed(self, mesh_model):
        model, calls = mesh_model
        (model / "mesh").mkdir()
        (model / "mesh" / "scan.ply").touch()
        results = MeshStage().run(ModelConfig.from_dir(model))
        assert calls == []
        assert [r.mesh_file.name for r in results] == ["scan.ply"]

    def test_each_geometry_source_meshed_by_its_own_name(self, mesh_model):
        # Remesh sources: CAD .step plus any tessellated surface format.
        model, calls = mesh_model
        (model / "geometry").mkdir()
        for name in ("a.step", "b.step", "raw.stl", "scan.ply", "head.obj"):
            (model / "geometry" / name).touch()
        results = MeshStage().run(ModelConfig.from_dir(model))
        assert calls == ["a.step", "b.step", "head.obj", "raw.stl", "scan.ply"]
        assert sorted(f.name for f in (model / "mesh").iterdir()) == \
            ["a.msh", "b.msh", "head.msh", "raw.msh", "scan.msh"]
        assert [r.mesh_file.name for r in results] == \
            ["a.msh", "b.msh", "head.msh", "raw.msh", "scan.msh"]

    def test_existing_msh_skips_that_source_only(self, mesh_model):
        model, calls = mesh_model
        (model / "geometry").mkdir()
        (model / "geometry" / "a.step").touch()
        (model / "geometry" / "b.step").touch()
        (model / "mesh").mkdir()
        (model / "mesh" / "a.msh").touch()
        MeshStage().run(ModelConfig.from_dir(model))
        assert calls == ["b.step"]

    def test_exec_script_can_import_bemkit(self, mesh_model, monkeypatch):
        # [geometry].exec scripts must import THIS bemkit even when the
        # package is not installed (externally-managed python) and cwd is
        # the model dir.
        model, _ = mesh_model
        monkeypatch.delenv("PYTHONPATH", raising=False)
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[geometry]\nexec = "./make_geo.py"\n')
        (model / "make_geo.py").write_text(
            "import bemkit, pathlib\n"
            "pathlib.Path('bemkit_path.txt').write_text(bemkit.__file__)\n")
        MeshStage().run(ModelConfig.from_dir(model))
        import bemkit
        assert (model / "bemkit_path.txt").read_text() == bemkit.__file__

    def test_pre_scripts_are_not_run(self, mesh_model, capsys):
        # The old pre_* glob is gone: leftovers are only reported, not run.
        model, _ = mesh_model
        (model / "pre_old.py").write_text(
            "import pathlib; pathlib.Path('ran.txt').touch()\n")
        MeshStage().run(ModelConfig.from_dir(model))
        assert not (model / "ran.txt").exists()
        assert "pre_old.py" in capsys.readouterr().out

    def test_missing_exec_script_raises(self, mesh_model):
        model, _ = mesh_model
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[geometry]\nexec = "./nope.py"\n')
        with pytest.raises(FileNotFoundError, match="nope.py"):
            MeshStage().run(ModelConfig.from_dir(model))

    def test_variant_mesh_overrides_generate_per_variant(self, mesh_model):
        # [variants] with mesh tables: one mesh per variant, named after it,
        # meshed from the sole geometry source with merged params.
        model, calls = mesh_model
        (model / "geometry").mkdir()
        (model / "geometry" / "box.step").touch()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[mesh]\nlc_min = 2.0\nlc_max = 4.0\n'
            '[variants.fine]\nmesh = { lc_min = 1.0 }\n'
            '[variants.coarse]\nmesh = { lc_min = 3.0 }\n')
        results = MeshStage().run(ModelConfig.from_dir(model))
        assert calls == ["box.step", "box.step"]
        assert sorted(f.name for f in (model / "mesh").iterdir()) == \
            ["coarse.msh", "fine.msh"]
        assert [r.mesh_file.name for r in results] == ["fine.msh", "coarse.msh"]

    def test_variant_existing_mesh_skipped(self, mesh_model):
        model, calls = mesh_model
        (model / "geometry").mkdir()
        (model / "geometry" / "box.step").touch()
        (model / "mesh").mkdir()
        (model / "mesh" / "fine.msh").touch()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[variants.fine]\nmesh = { lc_min = 1.0 }\n')
        results = MeshStage().run(ModelConfig.from_dir(model))
        assert calls == []
        assert [r.mesh_file.name for r in results] == ["fine.msh"]

    def test_variant_ambiguous_source_raises(self, mesh_model):
        # Two sources, no source= key, variant name matches neither.
        model, _ = mesh_model
        (model / "geometry").mkdir()
        (model / "geometry" / "a.step").touch()
        (model / "geometry" / "b.step").touch()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[variants.fine]\nmesh = { lc_min = 1.0 }\n')
        with pytest.raises(ValueError, match="source"):
            MeshStage().run(ModelConfig.from_dir(model))

    def test_variant_explicit_source(self, mesh_model):
        model, calls = mesh_model
        (model / "geometry").mkdir()
        (model / "geometry" / "a.step").touch()
        (model / "geometry" / "b.step").touch()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[variants.fine]\nsource = "b"\nmesh = { lc_min = 1.0 }\n')
        MeshStage().run(ModelConfig.from_dir(model))
        assert calls == ["b.step"]

    def test_variant_trusted_mesh_not_generated(self, mesh_model):
        # BC-defined variant (mesh = string): trusted file, never remeshed.
        model, calls = mesh_model
        (model / "mesh").mkdir()
        (model / "mesh" / "shared.msh").touch()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[variants.left]\nmesh = "shared"\n'
            '[variants.right]\nmesh = "shared"\n')
        results = MeshStage().run(ModelConfig.from_dir(model))
        assert calls == []
        assert [r.mesh_file.name for r in results] == ["shared.msh", "shared.msh"]


class TestExportStage:
    def test_fieldpoints_exec_runs_before_export(self, tmp_path):
        # The [fieldpoints].exec script runs at export time (even though the
        # stage then fails here for lack of meshes, the script has run).
        from bemkit.pipeline import ExportStage
        model = tmp_path / "m"
        model.mkdir()
        (model / "config.toml").write_text(
            '[bem]\nsr = 8000\nnbins = 5\n'
            '[fieldpoints]\nexec = "./make_fp.py"\n')
        (model / "make_fp.py").write_text(
            "import pathlib; pathlib.Path('fp_ran.txt').touch()\n")
        with pytest.raises(FileNotFoundError):
            ExportStage().run(ModelConfig.from_dir(model))
        assert (model / "fp_ran.txt").exists()


class TestRunStage:
    def test_force_clears_stale_npz_and_logs(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2])
        (nc / "NC3-4.out").write_text(
            "Step 3, Frequency = 750 Hz\n"
            "Warning: Maximum number of iterations is reached!\n")

        completed, failed = _stage_run(model_dir, force=True)

        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert not (nc / "be.npz").exists()
        assert not (nc / "NC3-4.out").exists()  # stale log removed
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]

    def test_default_resumes_restoring_npz_and_keeping_logs(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2])
        (nc / "NC.out").write_text("Step 1, Frequency = 100 Hz\nfine\n")

        completed, failed = _stage_run(model_dir)

        assert failed == []
        assert not (nc / "be.npz").exists()
        assert (nc / "NC.out").exists()  # resume keeps earlier logs
        # only the gap was computed; bins 1-2 came back from the npz
        assert (nc / "args.txt").read_text().split() == \
            ["-istart", "3", "-iend", "4"]
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]

    def test_skips_variant_with_running_numcalc(self, model_dir, monkeypatch,
                                                capsys):
        # A second NumCalc in the same variant dir would race the first
        # one's be.out/ writes — `run` must skip live variants.
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [4242])
        nc = model_dir / "numcalc" / "tetra"

        completed, failed = _stage_run(model_dir)

        assert (completed, failed) == ([], [])
        assert "already running" in capsys.readouterr().out
        assert not (nc / "args.txt").exists()  # NumCalc never invoked

    def test_complete_run_is_noop_by_default(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2, 3, 4])

        completed, failed = _stage_run(model_dir)

        assert (completed, failed) == ([], [])
        assert (nc / "be.npz").exists()        # results untouched
        assert not (nc / "args.txt").exists()  # NumCalc never invoked

    def test_force_allows_full_rerun_over_complete(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2, 3, 4])
        completed, failed = _stage_run(model_dir, force=True)
        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert not (nc / "be.npz").exists()          # moved aside
        assert (nc / "be.npz.bak").exists()          # backup kept
        assert (nc / "args.txt").exists()            # NumCalc ran

    def test_jobs_flag_fans_one_variant_over_bin_ranges(self, model_dir):
        # One variant, four slots: -j must not leave three processes idle.
        nc = model_dir / "numcalc" / "tetra"

        completed, failed = _stage_run(model_dir, max_jobs=4)

        assert len(completed) == 4 and failed == []
        assert sorted((nc / "args.txt").read_text().split("\n")[:-1]) == [
            "-istart 1 -iend 1", "-istart 2 -iend 2",
            "-istart 3 -iend 3", "-istart 4 -iend 4"]
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]

    def test_zero_jobs_rejected(self, model_dir):
        # -j 0 would otherwise read as "no limit" and start everything at once.
        with pytest.raises(ValueError, match="-j"):
            _stage_run(model_dir, max_jobs=0)
        assert not (model_dir / "numcalc" / "tetra" / "args.txt").exists()

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
        from bemkit.config import ModelConfig
        from bemkit.pipeline import PackStage

        model = tmp_path / "m"
        (model / "mesh").mkdir(parents=True)
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        for stem in ("good", "empty"):
            meshio.write_points_cells(model / "mesh" / f"{stem}.stl", verts,
                                      [("triangle", tris)])
        (model / "config.toml").write_text("[bem]\nsr = 8000\nnbins = 5\n")

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
