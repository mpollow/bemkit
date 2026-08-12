"""The `bem` CLI over the model.py spine.

Two kinds of study are used: a four-bin synthetic one written into tmp_path
(fast, and the only way to fake bin state without a solver) and a copy of
models/sphcap (real meshes, a real solve). Solving costs real time, so the
one end-to-end test computes a single bin of the coarsest variant.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bemkit.cli import _commands, main

REPO = Path(__file__).resolve().parents[4]
SPHCAP = REPO / "models" / "sphcap"

needs_numcalc = pytest.mark.skipif(
    shutil.which("NumCalc") is None, reason="NumCalc binary not on PATH")


# ── fixtures ───────────────────────────────────────────────────────────────

TINY_MODEL = """\
import numpy as np
from bemkit.model import BC, Elements, Mesh, Model, Velocity

VERTS = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
TRIS = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
AMPLITUDE = {amplitude}


def build():
    return Model(Mesh(VERTS, TRIS),
                 np.array([100., 200., 300., 400.]),
                 bcs=[BC(Elements([0]), Velocity(AMPLITUDE))],
                 fields={{"pts": np.array([[5., 5., 5.]])}})
"""


def write_study(path, amplitude=1.0):
    """A one-variant, four-bin study. Returns its directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.py").write_text(TINY_MODEL.format(amplitude=amplitude))
    return path


@pytest.fixture()
def study(tmp_path):
    return write_study(tmp_path / "tiny")


@pytest.fixture()
def sphcap(tmp_path):
    """A private copy, so a solve never writes into the repo's models/."""
    dst = tmp_path / "sphcap"
    shutil.copytree(SPHCAP, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "runs"))
    return dst


def fake_bins(wd, bins, sets=True):
    """Make ``available_bins(wd)`` report ``bins``.

    A bin only counts as complete when every expected quantity has content
    (``pEvalGrid`` too, whenever the bundle has an EvalGrid/), so both files
    are written — otherwise this helper would fake bins the real code drops.
    """
    wd.mkdir(parents=True, exist_ok=True)
    if sets:
        (wd / "EvalGrid").mkdir(exist_ok=True)
        (wd / "EvalGrid" / "Nodes.txt").write_text("1\n0 5 5 5\n")
    for b in bins:
        d = wd / "be.out" / f"be.{b}"
        d.mkdir(parents=True, exist_ok=True)
        for q in ("pBoundary", "pEvalGrid"):
            (d / q).write_text("h1\nh2\nh3\n0 1.0 2.0\n")


def solve_args(**over):
    """The Namespace `cmd_solve`/`_detach_solve` expect, with run defaults."""
    base = dict(variant=None, freq=None, bins=None, preview=None, jobs=1,
                force=False, foreground=False, restart=False, nitermax=None,
                check_normals=False, adapt_fmmlength=False)
    base.update(over)
    return argparse.Namespace(**base)


# ── status ─────────────────────────────────────────────────────────────────


def test_status_on_an_unsolved_study(study, capsys):
    main([str(study)])
    out = capsys.readouterr().out
    assert "variants: default" in out
    assert "export  ✗" in out
    assert "solve   ✗" in out
    assert "bins" not in out          # nothing computed: no count to show
    assert f"Next: bem {study} solve" in out


def test_status_shows_a_partial_run_distinctly(study, capsys):
    """A partial run is neither done nor untouched; ✗ alone reads as failure."""
    wd = _commands.workdir(study, "default")
    (wd).mkdir(parents=True)
    (wd / "NC.inp").write_text("")
    fake_bins(wd, [1])

    main([str(study), "status"])
    out = capsys.readouterr().out
    assert "export  ✓" in out
    assert "solve   ◐  (1/4 bins)" in out
    assert "Next: bem" in out


def test_status_marks_a_complete_run_done(study, capsys):
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1, 2, 3, 4])
    main([str(study), "status"])
    out = capsys.readouterr().out
    assert "solve   ✓  (4/4 bins)" in out
    assert "All variants solved." in out
    assert "Next:" not in out


def test_status_reports_solver_warnings(study, capsys):
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    (wd / "NC.out").write_text(
        "Step 1, Frequency = 100 Hz\n"
        "Warning: Maximum number of iterations is reached!\n")
    main([str(study), "status"])
    assert "⚠ 1 solver warning(s): bins 1" in capsys.readouterr().out


def test_status_names_stale_run_dirs(study, capsys):
    """A renamed variant leaves its results behind; they are otherwise
    invisible, since nothing else lists runs/ by name."""
    old = _commands.workdir(study, "wasnamedthis")
    old.mkdir(parents=True)
    (old / "NC.inp").write_text("")
    main([str(study)])
    assert "Stale run dirs (no matching variant): wasnamedthis" in \
        capsys.readouterr().out


def test_status_without_a_model_py_exits_with_a_hint(tmp_path):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path)])
    assert "model.py" in str(e.value)
    assert e.value.code != 2          # our message, not an argparse usage error


def test_study_dir_defaults_to_cwd(tmp_path, monkeypatch, capsys):
    write_study(tmp_path / "here")
    monkeypatch.chdir(tmp_path / "here")
    main([])
    assert str((tmp_path / "here").resolve()) in capsys.readouterr().out


def test_study_dir_named_like_a_subcommand_needs_a_path_prefix(
        tmp_path, monkeypatch, capsys):
    """`bem check` is the subcommand even when ./check is a study; `bem
    ./check` disambiguates."""
    write_study(tmp_path / "check")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):       # the subcommand, targeting cwd
        main(["check"])
    main(["./check"])                     # the directory
    assert str((tmp_path / "check").resolve()) in capsys.readouterr().out


# ── band selection ─────────────────────────────────────────────────────────


class TestBandSelectionReporting:
    """A requested frequency is snapped to the grid, never honoured exactly,
    so the substitution has to be visible."""

    FREQS = np.linspace(0, 16000, 65)[1:]      # sphcap's grid

    def _report(self, capsys, **over):
        _commands._selected_bins(self.FREQS, solve_args(**over))
        return capsys.readouterr().out

    def test_freq_reports_requested_and_actual(self, capsys):
        out = self._report(capsys, freq=[1100.0, 4000.0])
        assert "1100 Hz → bin 4 (1000.0 Hz)" in out
        assert "4000 Hz → bin 16 (4000.0 Hz)" in out
        assert "Partial band: 2 of 64 bins" in out

    def test_two_requests_landing_on_one_bin_say_so(self, capsys):
        out = self._report(capsys, freq=[990.0, 1010.0])
        assert "990 and 1010 Hz both resolve to bin 4 (1000.0 Hz)" in out

    def test_three_requests_landing_on_one_bin_say_all(self, capsys):
        out = self._report(capsys, freq=[990.0, 1000.0, 1010.0])
        assert "990 and 1000 and 1010 Hz all resolve to bin 4" in out

    def test_bins_are_listed_with_their_frequencies(self, capsys):
        out = self._report(capsys, bins=[4, 16])
        assert "1000.0 Hz (bin 4), 4000.0 Hz (bin 16)" in out

    def test_a_long_list_is_wrapped(self, capsys):
        out = self._report(capsys, preview=40)
        body = [ln for ln in out.splitlines() if ln.startswith("    ")]
        assert len(body) > 1                       # actually wrapped
        assert max(len(ln) for ln in body) <= 76

    def test_no_selector_means_the_whole_band(self, capsys):
        assert _commands._selected_bins(self.FREQS, solve_args()) is None
        assert capsys.readouterr().out == ""

    def test_an_impossible_selection_exits_with_the_reason(self):
        with pytest.raises(SystemExit) as e:
            _commands._selected_bins(self.FREQS, solve_args(bins=[999]))
        assert "outside the band" in str(e.value)


def test_freq_selection_prints_the_bin_through_the_cli(study, capsys, monkeypatch):
    """End to end, not just the helper: `solve` must report the snap before
    it does anything else."""
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(study), "solve", "--freq", "250"])
    out = capsys.readouterr().out
    # The grid is 100/200/300/400 Hz; 250 is a tie, and ties resolve low.
    assert "250 Hz → bin 2 (200.0 Hz)" in out


def test_variants_sharing_a_grid_report_once(sphcap, capsys, monkeypatch):
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(sphcap), "solve", "--bins", "1"])
    out = capsys.readouterr().out
    assert out.count("Partial band") == 1           # not once per variant


# ── --force refuses a partial band ─────────────────────────────────────────


@pytest.mark.parametrize("selector", [
    ["--freq", "250"], ["--bins", "2"], ["--preview", "2"],
])
def test_force_with_a_selector_exits(study, selector):
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve", "--force", *selector])
    # sys.exit(msg) is this file's convention: the message is the exit code
    # object, which the interpreter prints to stderr at top level.
    assert "--force" in str(e.value)
    assert "previous model" in str(e.value)
    assert "Full recompute:" in str(e.value)
    assert "Fresh partial:" in str(e.value)


def test_force_alone_is_accepted(study, monkeypatch, capsys):
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(study), "solve", "--force"])
    assert "Started in background" in capsys.readouterr().out


# ── resume ─────────────────────────────────────────────────────────────────


def test_a_complete_run_is_a_no_op(study, capsys):
    fake_bins(_commands.workdir(study, "default"), [1, 2, 3, 4])
    main([str(study), "solve", "--foreground"])
    assert "Nothing to run (all bins present" in capsys.readouterr().out


def test_a_present_selected_bin_is_not_recomputed(study, capsys):
    fake_bins(_commands.workdir(study, "default"), [2])
    main([str(study), "solve", "--bins", "2", "--foreground"])
    assert "all 1 selected bin(s) already present" in capsys.readouterr().out


def test_resume_schedules_only_the_missing_bins(study, monkeypatch, capsys):
    """The solver takes ``bins`` literally, so the resume arithmetic is the
    CLI's job — if it forwarded the whole band, bin 2 would be recomputed."""
    fake_bins(_commands.workdir(study, "default"), [2])
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    main([str(study), "solve", "--foreground"])
    assert seen["bins"] == [1, 3, 4]


def test_solve_saves_the_results(study, monkeypatch):
    """The CLI saves the Results itself.

    Asserted through a stub solver on purpose: the real ingest
    (``Results.from_beout``) already writes the same results.npz as a side
    effect, so a file-exists check after a real solve passes whether or not
    the CLI ever calls save — which is exactly the vacuous test this
    replaces.
    """
    wd = _commands.workdir(study, "default")
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver({}))
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    main([str(study), "solve", "--foreground"])
    assert (wd / "results.npz").exists()


def test_force_recomputes_everything_and_clears_be_out(study, monkeypatch):
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [2])
    (wd / "NC.out").write_text("stale warnings live here\n")
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    main([str(study), "solve", "--force", "--foreground"])
    # A full fresh band goes to NumCalc unranged (bins=None), which is what
    # keeps its plain NC.out invocation.
    assert seen["bins"] is None
    assert not (wd / "be.out" / "be.2").exists()
    assert not (wd / "NC.out").exists()      # stale log gone with its warnings


# ── a live NumCalc aborts the run ──────────────────────────────────────────


def _pretend_running(monkeypatch, wd, pids=(4321,)):
    """Make pynumcalc's /proc scan report `wd` busy, through the real
    running_variants -- patching _busy would test the patch."""
    import pynumcalc.status as status
    monkeypatch.setattr(status, "numcalc_pids",
                        lambda d: list(pids) if Path(d) == wd else [])


def test_a_live_numcalc_aborts_the_solve(study, monkeypatch):
    wd = _commands.workdir(study, "default")
    _pretend_running(monkeypatch, wd)
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve"])
    assert "already running" in str(e.value) and "4321" in str(e.value)
    assert "--restart" in str(e.value)


def test_force_is_not_an_override_for_a_live_numcalc(study, monkeypatch):
    """--force rmtree's be.out/; doing that under a live solver is the one
    combination that must stay impossible."""
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    _pretend_running(monkeypatch, wd)
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve", "--force", "--foreground"])
    assert "already running" in str(e.value)
    assert (wd / "be.out" / "be.1").exists()      # not discarded


def test_restart_kills_and_then_proceeds(study, monkeypatch, capsys):
    wd = _commands.workdir(study, "default")
    killed = []
    calls = {"n": 0}

    import pynumcalc.status as status

    def pids(d):
        # Busy on the pre-flight and the first _solve_foreground look, idle
        # once _kill_numcalc has run.
        if Path(d) != wd:
            return []
        calls["n"] += 1
        return [] if killed else [4321]

    monkeypatch.setattr(status, "numcalc_pids", pids)
    monkeypatch.setattr(_commands, "_kill_numcalc",
                        lambda p, **kw: killed.extend(p))
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))

    main([str(study), "solve", "--restart", "--foreground"])
    assert killed == [4321]
    assert seen["bins"] is None                   # it went on to solve
    assert "killing NumCalc (pid 4321)" in capsys.readouterr().out


# ── the changed-model guard ────────────────────────────────────────────────


class TestModelDigest:
    """The guard is only as good as the digest; a constant would pass every
    "unchanged" test and silently never fire."""

    def _model(self, **over):
        from bemkit.model import (
            BC, Elements, Medium, Mesh, Model, PointSource, Velocity)
        verts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        tris = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
        kw = dict(
            mesh=Mesh(over.pop("verts", verts), tris),
            frequencies=np.array([100., 200., 300., 400.]),
            medium=Medium(c=340.0, rho=1.2),
            bcs=[BC(Elements([0]), Velocity(1.0))],
            sources=[PointSource((9., 0., 0.))],
            fields={"a": np.array([[5., 5., 5.]]),
                    "b": np.array([[6., 6., 6.]])},
            exterior=True,
        )
        kw.update(over)
        mesh = kw.pop("mesh")
        freqs = kw.pop("frequencies")
        return Model(mesh, freqs, kw.pop("medium"), **kw)

    def test_the_same_model_hashes_the_same(self):
        assert (_commands._model_digest(self._model())
                == _commands._model_digest(self._model()))

    @pytest.mark.parametrize("name, change", [
        ("mesh", lambda s: dict(verts=np.array(
            [[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))),
        ("band", lambda s: dict(frequencies=np.array([100., 200., 300., 500.]))),
        ("medium", lambda s: dict(
            medium=__import__("bemkit.model", fromlist=["Medium"]).Medium(
                c=343.0, rho=1.2))),
        ("bc value", lambda s: dict(bcs=[s._bc(2.0)])),
        ("bc elements", lambda s: dict(bcs=[s._bc(1.0, elements=[1])])),
        ("source position", lambda s: dict(sources=[s._src((9., 1., 0.))])),
        ("source amplitude", lambda s: dict(sources=[s._src((9., 0., 0.), 2.0)])),
        ("field points", lambda s: dict(
            fields={"a": np.array([[7., 5., 5.]]),
                    "b": np.array([[6., 6., 6.]])})),
        ("field set order", lambda s: dict(
            fields={"b": np.array([[6., 6., 6.]]),
                    "a": np.array([[5., 5., 5.]])})),
        ("field set name", lambda s: dict(
            fields={"a": np.array([[5., 5., 5.]]),
                    "renamed": np.array([[6., 6., 6.]])})),
        ("domain", lambda s: dict(exterior=False)),
        ("time convention", lambda s: dict(time_factor=1.0)),
    ])
    def test_a_change_that_reaches_the_deck_changes_the_digest(self, name, change):
        base = _commands._model_digest(self._model())
        assert _commands._model_digest(self._model(**change(self))) != base, name

    def _bc(self, value, elements=(0,)):
        from bemkit.model import BC, Elements, Velocity
        return BC(Elements(list(elements)), Velocity(value))

    def _src(self, pos, amplitude=1.0):
        from bemkit.model import PointSource
        return PointSource(pos, amplitude)


def test_resuming_a_changed_model_is_refused(study, monkeypatch):
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    (wd / "model.sha256").write_text("a digest from some other model\n")
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve", "--foreground"])
    assert "model.py has changed" in str(e.value)
    assert "solve --force" in str(e.value)
    assert (wd / "be.out" / "be.1").exists()     # refused before touching it


def test_force_overrides_the_changed_model_guard(study, monkeypatch):
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    (wd / "model.sha256").write_text("a digest from some other model\n")
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))
    main([str(study), "solve", "--force", "--foreground"])
    assert seen["bins"] is None


def test_an_unchanged_model_resumes(study, monkeypatch):
    from bemkit.study import load_study
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    digest = _commands._model_digest(load_study(study)["default"])
    (wd / "model.sha256").write_text(digest + "\n")
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))
    main([str(study), "solve", "--foreground"])
    assert seen["bins"] == [2, 3, 4]


def test_bins_with_no_fingerprint_only_warn(study, monkeypatch, capsys):
    """Bundles from a hand-run NumCalc have no fingerprint; that is a reason
    to say so, not to refuse."""
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    seen = {}
    monkeypatch.setattr(_commands, "solver_for",
                        lambda *a, **k: _RecordingSolver(seen))
    main([str(study), "solve", "--foreground"])
    assert "no recorded model fingerprint" in capsys.readouterr().out
    assert seen["bins"] == [2, 3, 4]


def test_the_fingerprint_is_written_before_the_solve(study, monkeypatch):
    """A crash mid-solve still leaves bins behind, so the fingerprint that
    describes them must already be on disk."""
    from bemkit.study import load_study
    wd = _commands.workdir(study, "default")
    found = {}

    class Exploding:
        name = "numcalc"

        def solve(self, model, **kw):
            found["digest"] = (wd / "model.sha256").read_text().strip()
            raise RuntimeError("NumCalc fell over")

    monkeypatch.setattr(_commands, "solver_for", lambda *a, **k: Exploding())
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    with pytest.raises(SystemExit):
        main([str(study), "solve", "--foreground"])
    assert found["digest"] == _commands._model_digest(load_study(study)["default"])


# ── detaching ──────────────────────────────────────────────────────────────


class _FakePopen:
    pid = 4242
    last = None

    def __init__(self, argv, **kwargs):
        type(self).last = (argv, kwargs)


class _RecordingSolver:
    """Stands in for NumCalcSolver, capturing what the CLI asked it to do."""

    name = "numcalc"

    def __init__(self, seen):
        self.seen = seen

    def solve(self, model, *, workdir=None, bins=None, jobs=1, options=()):
        self.seen.update(bins=bins, jobs=jobs, options=options,
                         workdir=workdir)
        return _FakeResults()

    def export(self, model, workdir):
        return {}


class _FakeResults:
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        return Path(path)


def test_detach_builds_a_foreground_reinvocation(study, monkeypatch, capsys):
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)

    main([str(study), "solve", "-j", "3", "--nitermax", "5000",
          "--check-normals", "--adapt-fmmlength"])

    argv, kwargs = _FakePopen.last
    assert argv[1:3] == ["-m", "bemkit.cli"]
    assert argv[3] == str(study)
    assert argv[4] == "solve"
    assert "--foreground" in argv
    assert argv[argv.index("-j") + 1] == "3"
    assert argv[argv.index("--nitermax") + 1] == "5000"
    assert "--check-normals" in argv and "--adapt-fmmlength" in argv
    # Its own session, or the run dies with the terminal that started it.
    assert kwargs["start_new_session"] is True
    out = capsys.readouterr().out
    assert "Started in background (pid 4242)" in out
    assert str(study / "runs" / "run.log") in out
    assert (study / "runs" / "run.log").exists()


@pytest.mark.parametrize("selector, expected", [
    (["--preview", "2"], ["--preview", "2"]),
    (["--freq", "100", "400"], ["--freq", "100.0", "400.0"]),
    (["--bins", "1", "3"], ["--bins", "1", "3"]),
])
def test_detach_forwards_the_band_selection(study, monkeypatch, selector,
                                            expected):
    """The background supervisor is this same CLI re-invoked, so it must be
    told which bins to run or it would silently compute the whole band.

    Parametrized over all three selectors: they are three sibling branches in
    _detach_solve's argv construction, and covering one proves nothing about
    the other two.
    """
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(study), "solve", *selector])
    argv, _ = _FakePopen.last
    flag = expected[0]
    i = argv.index(flag)
    assert argv[i + 1:i + len(expected)] == expected[1:]


@pytest.mark.parametrize("field, flag", [
    ("bins", "--bins"), ("freq", "--freq"), ("preview", "--preview"),
])
def test_detach_forwards_a_present_but_empty_selector(study, monkeypatch,
                                                      field, flag):
    """`is not None`, not truthiness.

    The supervisor re-resolves the selection from these flags, so a selector
    that is present but falsy must still be forwarded as present — dropping
    it would silently run the whole band instead of erroring on an empty
    selection. argparse's nargs="+" cannot produce this today, which is
    exactly why it needs a direct test: nothing else would notice the day a
    caller (or a future flag shape) can.
    """
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    empty = 0 if field == "preview" else []
    _commands._detach_solve(study, solve_args(**{field: empty}),
                            {"default": [1]})
    argv, _ = _FakePopen.last
    assert flag in argv


def test_detach_is_refused_before_forking_when_numcalc_is_missing(
        study, monkeypatch):
    """Everything decidable up front must reach the terminal, not a log
    nobody is watching."""
    monkeypatch.setattr(_commands.shutil, "which", lambda n: None)
    forked = []
    monkeypatch.setattr(_commands.subprocess, "Popen",
                        lambda *a, **k: forked.append(a))
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve"])
    assert "NumCalc not found on PATH" in str(e.value)
    assert not forked


def test_detach_does_not_fork_for_an_empty_job_list(study, monkeypatch, capsys):
    fake_bins(_commands.workdir(study, "default"), [1, 2, 3, 4])
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    forked = []
    monkeypatch.setattr(_commands.subprocess, "Popen",
                        lambda *a, **k: forked.append(a))
    main([str(study), "solve"])
    assert not forked
    assert "Nothing to run" in capsys.readouterr().out


# ── variant selection ──────────────────────────────────────────────────────


def test_variant_prefixes_filter(sphcap, capsys, monkeypatch):
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(sphcap), "solve", "--variant", "co", "fi"])
    argv, _ = _FakePopen.last
    i = argv.index("--variant")
    assert argv[i + 1:i + 3] == ["co", "fi"]
    assert "2 variant(s)" in capsys.readouterr().out


def test_an_unmatched_variant_names_what_exists(sphcap):
    with pytest.raises(SystemExit) as e:
        main([str(sphcap), "solve", "--variant", "nope"])
    assert "coarse" in str(e.value) and "nope" in str(e.value)


# ── ps ─────────────────────────────────────────────────────────────────────


def test_ps_with_nothing_running(monkeypatch, capsys):
    import pynumcalc.status as status
    monkeypatch.setattr(status, "scan_numcalc", lambda: {})
    main([".", "ps"])
    assert capsys.readouterr().out == "No NumCalc processes running.\n"


def test_ps_reads_the_total_from_the_study(study, monkeypatch, capsys):
    """The done/total column's total comes from model.py, one level up from
    the layout `bem` used to walk (runs/<solver>/<variant>)."""
    import pynumcalc.status as status
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1, 2])
    monkeypatch.setattr(status, "scan_numcalc", lambda: {wd: [111, 222]})
    main([".", "ps"])
    out = capsys.readouterr().out
    assert "1 study/studies, 2 process(es)" in out
    assert "tiny" in out and "default" in out
    assert "2/4 bins" in out
    assert "pid 111, 222" in out


def test_ps_falls_back_to_a_bare_count_for_an_unreadable_study(
        study, monkeypatch, capsys):
    import pynumcalc.status as status
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1, 2])
    (study / "model.py").write_text("raise RuntimeError('broken study')\n")
    monkeypatch.setattr(status, "scan_numcalc", lambda: {wd: [111]})
    main([".", "ps"])
    out = capsys.readouterr().out
    assert "2 bins" in out and "/4" not in out     # listing survived


def test_ps_shows_the_raw_path_for_a_hand_launched_run(tmp_path, monkeypatch,
                                                       capsys):
    import pynumcalc.status as status
    odd = tmp_path / "somewhere" / "else"
    fake_bins(odd, [1])
    monkeypatch.setattr(status, "scan_numcalc", lambda: {odd: [7]})
    main([".", "ps"])
    assert str(odd) in capsys.readouterr().out


# ── pack ───────────────────────────────────────────────────────────────────


def test_pack_skips_a_variant_with_no_results(study, capsys):
    main([str(study), "pack"])
    out = capsys.readouterr().out
    assert "skipping (no results)" in out and "Nothing packed." in out


def test_pack_skips_a_live_variant_rather_than_aborting(study, monkeypatch,
                                                        capsys):
    """Packing deletes be.out/ from under a solver still writing into it —
    but the finished variants should still pack, so this skips."""
    wd = _commands.workdir(study, "default")
    fake_bins(wd, [1])
    _pretend_running(monkeypatch, wd)
    main([str(study), "pack"])
    assert "skipping (NumCalc still running)" in capsys.readouterr().out
    assert (wd / "be.out" / "be.1").exists()


# ── jobs ───────────────────────────────────────────────────────────────────


def test_jobs_defaults_to_one(study, monkeypatch, capsys):
    monkeypatch.setattr(_commands.shutil, "which", lambda n: "/usr/bin/NumCalc")
    monkeypatch.setattr(_commands.subprocess, "Popen", _FakePopen)
    main([str(study), "solve"])
    argv, _ = _FakePopen.last
    assert argv[argv.index("-j") + 1] == "1"


def test_zero_jobs_is_refused(study):
    with pytest.raises(SystemExit) as e:
        main([str(study), "solve", "-j", "0"])
    assert "-j must be at least 1" in str(e.value)


# ── end to end ─────────────────────────────────────────────────────────────


@needs_numcalc
def test_solve_one_bin_end_to_end(sphcap, capsys):
    """One bin of the coarsest variant — a real NumCalc run, kept to the
    smallest thing that proves the whole path."""
    main([str(sphcap), "solve", "--variant", "coarse", "--bins", "1",
          "--foreground"])

    wd = sphcap / "runs" / "numcalc" / "coarse"
    assert (wd / "NC.inp").exists()
    assert (wd / "ObjectMesh" / "Nodes.txt").exists()
    assert (wd / "EvalGrid" / "Nodes.txt").exists()
    assert (wd / "results.npz").exists()
    assert (wd / "be.out" / "be.1").is_dir()
    assert "Done." in capsys.readouterr().out

    # The fingerprint recorded for the bins that were just computed.
    from bemkit.study import load_study
    assert (wd / "model.sha256").read_text().strip() == \
        _commands._model_digest(load_study(sphcap)["coarse"])

    # Real numbers, at the bin that was asked for and nowhere else.
    from bemkit.results import Results
    r = Results.load(wd / "results.npz")
    p = r.pressure("onaxis")
    assert np.isfinite(p[:, 0]).all()
    assert np.isnan(p[:, 1]).all()          # bin 2 was not requested

    # And status now agrees.
    main([str(sphcap)])
    out = capsys.readouterr().out
    assert "solve   ◐  (1/64 bins)" in out


@needs_numcalc
def test_a_second_solve_of_the_same_bin_is_a_no_op(sphcap, capsys):
    main([str(sphcap), "solve", "--variant", "coarse", "--bins", "1",
          "--foreground"])
    capsys.readouterr()
    main([str(sphcap), "solve", "--variant", "coarse", "--bins", "1",
          "--foreground"])
    assert "already present" in capsys.readouterr().out


@needs_numcalc
def test_estimate_ram_exports_the_bundle_itself(sphcap, capsys):
    """There is no `export` command any more, so demanding a solve first
    would mean running the very solve the estimate exists to size."""
    main([str(sphcap), "estimate-ram", "--variant", "coarse"])
    out = capsys.readouterr().out
    assert (sphcap / "runs" / "numcalc" / "coarse" / "NC.inp").exists()
    assert "coarse:" in out and "GB" in out
    assert "bin 64 (16000 Hz)" in out
    # Clustering only: no bin was solved (NumCalc still makes the be.out/
    # shell, so the claim has to be about its contents).
    from pynumcalc.status import available_bins
    assert available_bins(sphcap / "runs" / "numcalc" / "coarse") == []
