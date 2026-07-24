"""RunStage glue: be.npz clearing, stale-log removal, solver-warning report.

Uses a fake NumCalc that mimics the real binary's semantics: unranged runs
wipe be.out/ and log to NC.out, ranged runs preserve be.out/ and log to
NC{istart}-{iend}.out.
"""

import os
import shutil
import sys
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

    def test_aborts_when_numcalc_is_already_running(self, model_dir, monkeypatch):
        # A second NumCalc in the same variant dir would race the first one's
        # be.out/ writes. Quietly doing less than asked is the wrong answer for
        # a job this expensive — stop and name the pids.
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [4242])
        nc = model_dir / "numcalc" / "tetra"

        with pytest.raises(RuntimeError, match="4242"):
            _stage_run(model_dir)

        assert not (nc / "args.txt").exists()  # NumCalc never invoked

    def test_abort_message_names_the_variant_and_points_at_restart(
            self, model_dir, monkeypatch):
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [4242])

        with pytest.raises(RuntimeError) as exc:
            _stage_run(model_dir)

        assert "tetra" in str(exc.value) and "--restart" in str(exc.value)

    def test_force_still_aborts_on_a_live_variant(self, model_dir, monkeypatch):
        """--force rmtree's be.out/. Doing that under a running solver is the
        one combination that must stay impossible, so --force is not an
        override for this check."""
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [4242])
        nc = model_dir / "numcalc" / "tetra"
        (nc / "be.out" / "be.1").mkdir(parents=True)

        with pytest.raises(RuntimeError, match="4242"):
            _stage_run(model_dir, force=True)

        assert (nc / "be.out" / "be.1").exists()  # not deleted

    def test_restart_kills_the_running_processes_then_runs(self, model_dir,
                                                           monkeypatch):
        import bemkit.sim.status as status
        import bemkit.pipeline as pipeline
        live = {4242}
        killed = []
        monkeypatch.setattr(status, "numcalc_pids",
                            lambda d: sorted(live))

        def fake_kill(pids):
            killed.extend(pids)
            live.clear()

        monkeypatch.setattr(pipeline, "_kill_numcalc", fake_kill)
        nc = model_dir / "numcalc" / "tetra"

        completed, failed = _stage_run(model_dir, restart=True)

        assert killed == [4242]
        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert (nc / "args.txt").exists()  # NumCalc did run afterwards

    def test_restart_resumes_rather_than_recomputing(self, model_dir,
                                                     monkeypatch):
        """Killing leaves a half-written bin, which complete_bins already
        rejects — so a plain --restart can safely resume the gap. Wiping
        everything stays opt-in via --force."""
        import bemkit.sim.status as status
        import bemkit.pipeline as pipeline
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [])
        monkeypatch.setattr(pipeline, "_kill_numcalc", lambda pids: None)
        nc = model_dir / "numcalc" / "tetra"
        _make_npz(nc, [1, 2])

        _stage_run(model_dir, restart=True)

        assert (nc / "args.txt").read_text().split() == \
            ["-istart", "3", "-iend", "4"]

    def test_restart_without_anything_running_just_runs(self, model_dir):
        nc = model_dir / "numcalc" / "tetra"

        completed, failed = _stage_run(model_dir, restart=True)

        assert [j.name for j in completed] == ["tetra"] and failed == []
        assert (nc / "args.txt").exists()

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

    def test_force_clears_be_out_even_when_fanned_out(self, model_dir):
        # NumCalc only wipes be.out/ for an unranged invocation, so once -j
        # exceeds the variant count and --force fans out into bin ranges,
        # nothing would clear bins left by an earlier run over a different
        # eval grid — they would silently mix into the new results.
        nc = model_dir / "numcalc" / "tetra"
        (nc / "be.out" / "be.9").mkdir(parents=True)

        completed, failed = _stage_run(model_dir, force=True, max_jobs=4)

        assert len(completed) == 4 and failed == []
        assert sorted(p.name for p in (nc / "be.out").iterdir()) == \
            ["be.1", "be.2", "be.3", "be.4"]  # stale be.9 gone

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


def _run_args(argv):
    """Parse `run` flags through the real CLI parser, so these tests break if
    the flag surface drifts."""
    import argparse
    from bemkit.cli import _pipeline
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _pipeline.register(sub)
    return parser.parse_args(["run", *argv])


class _FakePopen:
    """Records the arguments cmd_run detaches with."""

    calls = []

    def __init__(self, argv, **kw):
        self.pid = 4242
        _FakePopen.calls.append((argv, kw))


@pytest.fixture()
def spawned(monkeypatch):
    """Intercept only the CLI's own Popen — patching subprocess.Popen itself
    would also catch LocalRunner submitting NumCalc, which the --foreground
    test needs to work for real."""
    import subprocess as real

    class _Shim:
        Popen = _FakePopen
        DEVNULL = real.DEVNULL
        STDOUT = real.STDOUT

    from bemkit.cli import _pipeline
    _FakePopen.calls = []
    monkeypatch.setattr(_pipeline, "subprocess", _Shim)
    return _FakePopen.calls


class TestRunCommandBackground:
    """`bem run` detaches and returns; the supervisor it starts is this same
    CLI re-invoked with --foreground."""

    def _cmd_run(self, model_dir, argv=()):
        from bemkit.cli._pipeline import cmd_run
        return cmd_run(model_dir, _run_args(argv))

    def test_detaches_instead_of_running_numcalc_inline(self, model_dir, spawned):
        self._cmd_run(model_dir)

        assert len(spawned) == 1
        assert not (model_dir / "numcalc" / "tetra" / "args.txt").exists()

    def test_supervisor_is_started_in_its_own_session(self, model_dir, spawned):
        """Without a new session the run dies with the terminal that spawned
        it — the whole point is to outlive the shell."""
        self._cmd_run(model_dir)
        _, kw = spawned[0]

        assert kw["start_new_session"] is True

    def test_supervisor_runs_in_the_foreground(self, model_dir, spawned):
        """--foreground is also the recursion guard: without it the detached
        copy would fork again, forever."""
        self._cmd_run(model_dir)
        argv, _ = spawned[0]

        assert "--foreground" in argv
        assert argv[:3] == [sys.executable, "-m", "bemkit.cli"]
        assert str(model_dir) in argv and "run" in argv

    def test_forwards_the_run_flags_to_the_supervisor(self, model_dir, spawned):
        self._cmd_run(model_dir, ["-j", "3", "--force", "--variant", "tetra"])
        argv, _ = spawned[0]

        assert argv[argv.index("-j") + 1] == "3"
        assert "--force" in argv
        assert argv[argv.index("--variant") + 1] == "tetra"

    def test_writes_a_log_next_to_the_variant_bundles(self, model_dir, spawned,
                                                      capsys):
        self._cmd_run(model_dir)
        log = model_dir / "numcalc" / "run.log"

        assert log.exists()
        assert str(log) in capsys.readouterr().out

    def test_foreground_runs_inline_and_spawns_nothing(self, model_dir, spawned):
        self._cmd_run(model_dir, ["--foreground"])

        assert spawned == []
        assert (model_dir / "numcalc" / "tetra" / "args.txt").exists()

    def test_a_busy_variant_stops_it_before_detaching(self, model_dir, spawned,
                                                     monkeypatch):
        """The abort must land on the terminal, not in a log file nobody is
        watching yet."""
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [99])

        with pytest.raises(SystemExit, match="99"):
            self._cmd_run(model_dir)

        assert spawned == []

    def test_missing_numcalc_stops_it_before_detaching(self, model_dir, spawned,
                                                      monkeypatch):
        monkeypatch.setenv("PATH", "")

        with pytest.raises(SystemExit, match="NumCalc"):
            self._cmd_run(model_dir)

        assert spawned == []

    def test_nothing_to_run_does_not_detach(self, model_dir, spawned, capsys):
        """Spawning a supervisor that immediately finds no work would report
        it into a log the user has no reason to open."""
        _make_npz(model_dir / "numcalc" / "tetra", [1, 2, 3, 4])

        self._cmd_run(model_dir)

        assert spawned == []
        assert "Nothing to run" in capsys.readouterr().out


class TestPsCommand:
    """`bem ps` — the one cross-model view: which project NumCalc is on right
    now, without having to already know which model to ask about."""

    def _cmd_ps(self, model_dir):
        import argparse
        from bemkit.cli._pipeline import cmd_ps
        return cmd_ps(model_dir, argparse.Namespace())

    def test_reports_model_variant_progress_and_pid(self, model_dir,
                                                    monkeypatch, capsys):
        import bemkit.sim.status as status
        nc = model_dir / "numcalc" / "tetra"
        monkeypatch.setattr(status, "scan_numcalc", lambda: {nc: [4242]})
        monkeypatch.setattr(status, "available_bins", lambda d: [1, 2])

        self._cmd_ps(model_dir)
        out = capsys.readouterr().out

        assert model_dir.name in out         # the project
        assert "tetra" in out                # the variant
        assert "2/4 bins" in out             # progress against the model's nf
        assert "4242" in out                 # the pid

    def test_reports_process_count_when_a_variant_is_fanned_out(self, model_dir,
                                                               monkeypatch, capsys):
        import bemkit.sim.status as status
        nc = model_dir / "numcalc" / "tetra"
        monkeypatch.setattr(status, "scan_numcalc", lambda: {nc: [10, 11]})
        monkeypatch.setattr(status, "available_bins", lambda d: [1])

        self._cmd_ps(model_dir)
        out = capsys.readouterr().out

        assert "10, 11" in out
        assert "2 proc" in out

    def test_says_so_when_nothing_is_running(self, model_dir, monkeypatch, capsys):
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "scan_numcalc", lambda: {})

        self._cmd_ps(model_dir)

        assert "No NumCalc" in capsys.readouterr().out

    def test_survives_a_process_outside_the_standard_layout(self, tmp_path,
                                                           monkeypatch, capsys):
        """A NumCalc launched by hand somewhere odd has no model/numcalc/variant
        path to parse — show it rather than crashing the whole listing."""
        import bemkit.sim.status as status
        odd = tmp_path / "somewhere"
        monkeypatch.setattr(status, "scan_numcalc", lambda: {odd: [7]})
        monkeypatch.setattr(status, "available_bins", lambda d: [])

        self._cmd_ps(tmp_path)
        out = capsys.readouterr().out

        assert "7" in out and "somewhere" in out


class TestPackStageDeletesByDefault:
    """be.out/ is >99 % of a variant directory and the pack is verified bin by
    bin against it before anything is removed, so keeping it is the exception."""

    @pytest.fixture()
    def packable(self, tmp_path):
        """One-variant model with two real bins in be.out/."""
        import meshio
        model = tmp_path / "m"
        (model / "mesh").mkdir(parents=True)
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        tris = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        meshio.write_points_cells(model / "mesh" / "v.stl", verts,
                                  [("triangle", tris)])
        (model / "config.toml").write_text("[bem]\nsr = 8000\nnbins = 5\n")
        for bi in (1, 2):
            d = model / "numcalc" / "v" / "be.out" / f"be.{bi}"
            d.mkdir(parents=True)
            for q in ("pBoundary", "pEvalGrid"):
                rows = "".join(f"{j} {bi*10+j}.0 {bi*100+j}.0\n" for j in range(3))
                (d / q).write_text("h1\nh2\nh3\n" + rows)
        return model

    def _pack(self, model, **kw):
        from bemkit.pipeline import PackStage
        return PackStage().run(ModelConfig.from_dir(model), **kw)

    def test_bare_pack_removes_be_out(self, packable):
        self._pack(packable)

        assert (packable / "numcalc" / "v" / "be.npz").exists()
        assert not (packable / "numcalc" / "v" / "be.out").exists()

    def test_keep_original_retains_be_out(self, packable):
        self._pack(packable, delete_original=False)

        assert (packable / "numcalc" / "v" / "be.npz").exists()
        assert (packable / "numcalc" / "v" / "be.out").is_dir()

    def test_deleted_be_out_can_be_restored_from_the_pack(self, packable):
        """What makes the default safe: the removal is reversible."""
        from bemkit.sim.packing import unpack_variant
        self._pack(packable)

        unpack_variant(packable / "numcalc" / "v", verbose=False)

        assert sorted(p.name for p in
                      (packable / "numcalc" / "v" / "be.out").iterdir()) == \
            ["be.1", "be.2"]

    def test_skips_a_variant_numcalc_is_still_writing(self, packable,
                                                      monkeypatch, capsys):
        """Deleting be.out/ under a live solver would destroy the bins it is
        in the middle of writing. Skip that variant, don't abort the model."""
        import bemkit.sim.status as status
        monkeypatch.setattr(status, "numcalc_pids", lambda d: [777])

        packed = self._pack(packable)

        assert packed == []
        assert (packable / "numcalc" / "v" / "be.out").is_dir()
        assert not (packable / "numcalc" / "v" / "be.npz").exists()
        assert "777" in capsys.readouterr().out


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
