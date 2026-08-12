"""`bem` subcommands over the model.py spine.

The old pipeline's stage machine is gone: a study is Python, so there is
nothing to stage. What is left is the part of the CLI that was never about
config parsing — knowing when *not* to start a solver, what a partial band
resolved to, and which of your machine's NumCalc processes belongs to which
study.

Runs live in ``STUDY/runs/<solver>/<variant>/``. That directory is the
bundle: self-contained, relocatable, and resumable because frequency bins
are independent.
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Everything a run writes lives under this, one level per solver.
RUNS_DIR = "runs"


def register(sub):
    p = sub.add_parser("status", help="Per-variant export/solve state incl. "
                                      "running NumCalc jobs (the default)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("solve", help="Export the bundle and run NumCalc "
                                     "(computes missing bins only — resumes "
                                     "interrupted runs)")
    _add_variant(p)
    _add_band(p)
    p.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                   help="Max NumCalc processes at once (default: 1). "
                        "Variants are solved one after another; within a "
                        "variant the missing bins are fanned out into ranges "
                        "to fill N processes")
    p.add_argument("--force", action="store_true",
                   help="Recompute all bins from scratch, discarding existing "
                        "results (needed after model.py changed the mesh or "
                        "the frequency band)")
    p.add_argument("--foreground", action="store_true",
                   help="Wait for NumCalc instead of detaching (default is to "
                        "start in the background and return)")
    p.add_argument("--restart", action="store_true",
                   help="Kill NumCalc processes already running for the "
                        "selected variants, then run. Resumes missing bins; "
                        "add --force to recompute everything")
    p.add_argument("--nitermax", type=int, metavar="N",
                   help="Max CGS iterations before NumCalc gives up "
                        "(NumCalc default: 1000). Non-convergence usually "
                        "means a defective mesh — re-mesh before raising this")
    p.add_argument("--check-normals", action="store_true",
                   help="Have NumCalc verify all element normals point into "
                        "the same domain. Note it does NOT check they point "
                        "outward, so a uniformly inverted mesh still passes")
    p.add_argument("--adapt-fmmlength", action="store_true",
                   help="Adapt the multipole truncation to cluster radii "
                        "(one of the two levers when expansion order L > 30)")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("estimate-ram",
                       help="Run NumCalc's clustering only (no solve) and "
                            "report its RAM estimate per variant")
    _add_variant(p)
    p.add_argument("--all-bins", action="store_true",
                   help="Estimate across the whole band instead of only the "
                        "highest frequency (where FMM memory peaks)")
    p.set_defaults(func=cmd_estimate_ram)

    p = sub.add_parser("pack", help="Pack be.out/ into be.npz (~4x shrink) "
                                    "and remove be.out/")
    _add_variant(p)
    p.add_argument("--keep-original", action="store_true",
                   help="Keep be.out/ instead of removing it after the "
                        "verified round-trip")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("ps", help="List NumCalc processes running anywhere "
                                  "(study / variant / bins / pid). Ignores "
                                  "STUDY_DIR")
    p.set_defaults(func=cmd_ps)


def _add_variant(p):
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", metavar="PREFIX", default=None,
                   action=VariantAction,
                   help="Only process variants matching these prefixes "
                        "(repeatable, and space-separated)")


def _add_band(p):
    """Selectors for a partial-band run (mutually exclusive)."""
    g = p.add_mutually_exclusive_group()
    g.add_argument("-f", "--freq", nargs="+", type=float, metavar="HZ",
                   help="Run only the bins nearest these frequencies "
                        "(snapped to the model's frequency grid)")
    g.add_argument("--bins", nargs="+", type=int, metavar="N",
                   help="Run only these bins (1-based)")
    g.add_argument("--preview", type=int, metavar="N",
                   help="Run N bins spread logarithmically over the band — "
                        "a fast look at the model; a later run fills in the rest")


# ── study / workdir plumbing ────────────────────────────────────────────────


def solver_for(name="numcalc"):
    """The solver backend. One today; the name is already the runs/ level."""
    from bemkit.solvers.numcalc import NumCalcSolver
    if name != NumCalcSolver.name:
        sys.exit(f"unknown solver {name!r} (only 'numcalc' exists)")
    return NumCalcSolver()


def workdir(study_dir, variant, solver="numcalc"):
    """``STUDY/runs/<solver>/<variant>/`` — the bundle for one variant."""
    return Path(study_dir) / RUNS_DIR / solver / variant


def load(study_dir):
    """``load_study`` with a CLI-friendly error for the missing-study case.

    Anything model.py itself raises propagates with its traceback: a study is
    Python, and a KeyError in ``build()`` is a line number the author needs,
    not a message this layer could improve on.
    """
    from bemkit.cli.study import load_study
    try:
        return load_study(study_dir)
    except FileNotFoundError as e:
        sys.exit(f"{e}\n"
                 f"  A study is a directory holding a model.py that defines "
                 f"build() -> Model (or variants() -> {{name: Model}}).")


def select(models, prefixes):
    """Variant names matching ``prefixes`` (all of them when None)."""
    names = list(models)
    if not prefixes:
        return names
    chosen = [n for n in names if any(n.startswith(p) for p in prefixes)]
    if not chosen:
        sys.exit(f"No variant matches {', '.join(prefixes)} — this study has: "
                 f"{', '.join(names)}")
    return chosen


# ── band selection ─────────────────────────────────────────────────────────


def _has_band_selection(args):
    """True if --freq/--bins/--preview was given (any one of them).

    The one place this is asked, spelled the same way everywhere it is
    asked: an empty list (unreachable via argparse's ``nargs="+"``, but not
    via a direct caller) must count as "given" here exactly as it does in
    ``_selected_bins``/``_reject_force_with_selection``/``_detach_solve``'s
    argv forwarding — a truthiness check on the list would disagree.
    """
    return args.freq is not None or args.bins is not None or args.preview is not None


def _selected_bins(frequencies, args, label=""):
    """Resolve --freq/--bins/--preview to bin indices, or None for the full band.

    Prints what was resolved, with the actual bin frequencies, because a
    requested frequency is snapped to the grid rather than honoured exactly.
    ``label`` names the variants a grid belongs to when a study has more than
    one frequency grid.
    """
    from bemkit.model import resolve_bins

    if not _has_band_selection(args):
        return None

    try:
        bins = resolve_bins(frequencies, freq=args.freq, bins=args.bins,
                            preview=args.preview)
    except ValueError as e:
        sys.exit(str(e))

    freqs = np.asarray(frequencies, dtype=float)
    nf = len(freqs)
    print(f"  Partial band{label}: {len(bins)} of {nf} bins")

    if args.freq is not None:
        # One line per request, showing what it snapped to. A requested
        # frequency is almost never a bin frequency, and the substitution is
        # the thing the user needs to see — a bare bin list hides it.
        landed = {}
        for f in args.freq:
            b = resolve_bins(frequencies, freq=[f])[0]
            landed.setdefault(b, []).append(f)
        for b in sorted(landed):
            asked = landed[b]
            joined = " and ".join(f"{f:g}" for f in asked)
            if len(asked) > 1:
                every = "both" if len(asked) == 2 else "all"
                print(f"    {joined} Hz {every} resolve to bin {b} "
                      f"({freqs[b - 1]:.1f} Hz)")
            else:
                print(f"    {joined} Hz → bin {b} ({freqs[b - 1]:.1f} Hz)")
    else:
        # --bins/--preview name bins directly, so the list is the useful form
        # — wrapped, because a 64-bin selection is otherwise one long line.
        import textwrap
        listed = ", ".join(f"{freqs[b - 1]:.1f} Hz (bin {b})" for b in bins)
        print(textwrap.fill(listed, width=76, initial_indent="    ",
                            subsequent_indent="    "))
    return bins


def _bins_per_variant(models, names, args):
    """``{variant: selected bins or None}``.

    Variants are separate Models and need not share a frequency grid, so the
    selection is resolved against each one's own frequencies — but variants
    that *do* share a grid (the common case, e.g. one mesh resolution per
    variant) share one report, or the same three lines would print three
    times.
    """
    groups = {}
    for v in names:
        groups.setdefault(models[v].frequencies.tobytes(), []).append(v)

    out = {}
    for members in groups.values():
        label = ("" if len(groups) == 1
                 else f" ({', '.join(members)})")
        bins = _selected_bins(models[members[0]].frequencies, args, label)
        for v in members:
            out[v] = bins
    return out


def _reject_force_with_selection(args):
    """--force and a bin selection cannot be combined.

    --force discards every existing bin because bin presence says nothing
    about which model produced it. Scoping it to a subset would leave the
    unnamed bins in place — results from two different models in one
    be.out/, which is the corruption --force exists to prevent.
    """
    if not args.force:
        return
    if not _has_band_selection(args):
        return
    sys.exit(
        "--force discards every existing bin and cannot be combined with "
        "--freq/--bins/--preview: the bins you did not name would still "
        "come from the previous model.\n"
        "  Full recompute:  bem STUDY solve --force\n"
        "  Fresh partial:   rm -r runs/numcalc/<variant>/be.out  &&  "
        "bem STUDY solve --preview 8")


# ── the changed-model guard ────────────────────────────────────────────────
#
# A resume re-exports NC.inp from whatever model.py says *now* and then asks
# NumCalc for the bins that are missing. If model.py changed in between, the
# new bins answer a different problem than the old ones and be.out/ ends up
# holding two models' results with nothing to tell them apart. The old CLI
# had the same hazard and answered it with "use --force"; this makes the
# answer automatic by recording what produced the bins.


def _model_digest(model):
    """A stable hash of everything about ``model`` that reaches the deck.

    Covers the mesh, the band, the medium, the domain/time conventions, the
    resolved boundary conditions (elements *and* their values across the
    band), the sources, and the field-point sets in their emission order.
    That is the whole input to ``model_to_deck`` plus the EvalGrid.

    Maintenance note: a new Model attribute that changes what NumCalc reads
    must be added here, or a change to it will slip past the resume guard.
    The guard is an honesty check on resumes, not a correctness invariant —
    a miss costs a misleading merge, not a wrong answer to the deck actually
    solved.
    """
    from bemkit.model.sources import PlaneWave

    h = hashlib.sha256()

    def tag(*parts):
        for p in parts:
            h.update(repr(p).encode())
            h.update(b"\0")

    def arr(a, dtype):
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
        h.update(b"\0")

    tag("v1", model.exterior, model.time_factor,
        model.medium.c, model.medium.rho)
    arr(model.frequencies, float)
    arr(model.mesh.vertices, float)
    arr(model.mesh.triangles, np.int64)

    for rbc in model.resolved_bcs():
        cond = rbc.condition
        tag(type(cond).__name__, cond.kind)
        arr(np.sort(np.asarray(rbc.elements, dtype=np.int64)), np.int64)
        arr(np.atleast_1d(cond.at(model.frequencies)), complex)

    for s in model.sources:
        vec = s.direction if isinstance(s, PlaneWave) else s.position
        tag(type(s).__name__, vec, s.amplitude)

    # Dict order, not sorted: it is the order the sets are concatenated into
    # the EvalGrid, so two studies differing only in it differ in NC.inp.
    for name, pts in model.fields.items():
        tag(name)
        arr(pts, float)

    return h.hexdigest()


_DIGEST_FILE = "model.sha256"


def _guard_unchanged_model(wd, variant, digest, present, force):
    """Refuse to resume into bins a different model.py produced."""
    if force or not present:
        return                      # nothing on disk worth protecting
    path = wd / _DIGEST_FILE
    if not path.exists():
        print(f"  {variant}: warning — {len(present)} existing bin(s) with no "
              f"recorded model fingerprint, so they cannot be checked against "
              f"the current model.py. Use --force if in doubt.")
        return
    if path.read_text().strip() == digest:
        return
    sys.exit(
        f"{variant}: model.py has changed since the {len(present)} existing "
        f"bin(s) were computed ({wd / _DIGEST_FILE} does not match). "
        f"Resuming would mix two models' results in one be.out/.\n"
        f"  Recompute everything:  bem {wd.parents[2]} solve --force "
        f"--variant {variant}\n"
        f"  Or keep the old run:   move {wd} aside first")


# ── status ─────────────────────────────────────────────────────────────────


def cmd_status(study_dir, args):
    return show_status(study_dir)


def show_status(study_dir):
    """Default command: per-variant export/solve state + next-step hint."""
    from pynumcalc.status import (
        available_bins, current_step, numcalc_pids, solver_warnings)

    models = load(study_dir)
    solver = solver_for()
    names = list(models)
    if not names:
        print(f"\n{study_dir}  (model.py declares no variants)\n")
        return

    print(f"\n{study_dir}  (solver: {solver.name}; "
          f"variants: {', '.join(names)})\n")

    running, todo = [], []
    for v in names:
        wd = workdir(study_dir, v, solver.name)
        nf = len(models[v].frequencies)
        exported = (wd / "NC.inp").exists()
        present = available_bins(wd) if wd.is_dir() else []
        pids = numcalc_pids(wd)

        print(f"  {v}:")
        print(f"    export  {'✓' if exported else '✗'}")

        n = len(present)
        mark = "✓" if n >= nf else ("◐" if n else "✗")
        extra = f"  ({n}/{nf} bins)" if n else ""
        if pids:
            running.append(v)
            step = current_step(wd)
            at = f", computing bin {step}" if step else ""
            extra += f"  ⟳ running ({len(pids)} NumCalc process(es){at})"
        issues = solver_warnings(wd)
        if issues:
            bins = sorted({b for _, b, _ in issues if b is not None})
            where = f": bins {', '.join(map(str, bins))}" if bins else ""
            extra += f"  ⚠ {len(issues)} solver warning(s){where}"
        print(f"    solve   {mark}{extra}")
        if n < nf:
            todo.append(v)
        print()

    _print_stale(study_dir, solver.name, names)

    rel = (study_dir.relative_to(Path.cwd())
           if study_dir.is_relative_to(Path.cwd()) else study_dir)
    pending = [v for v in todo if v not in running]
    if not todo:
        print("  All variants solved.")
    elif not pending:
        print(f"  NumCalc running ({', '.join(running)}) — "
              f"re-run `bem` to refresh.")
    else:
        count = (f"  ({len(pending)}/{len(names)} variants)"
                 if len(names) > 1 else "")
        print(f"  Next: bem {rel} solve{count}")


def _print_stale(study_dir, solver_name, names):
    """Run directories with no matching variant — a renamed variant leaves
    its old results behind, and they are otherwise invisible."""
    root = Path(study_dir) / RUNS_DIR / solver_name
    if not root.is_dir():
        return
    known = set(names)
    stale = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and d.name not in known
                   and (d / "NC.inp").exists())
    if stale:
        print(f"  Stale run dirs (no matching variant): {', '.join(stale)}\n")


# ── solve ──────────────────────────────────────────────────────────────────


def cmd_solve(study_dir, args):
    from pynumcalc.runner import solver_flags
    from pynumcalc.status import available_bins

    models = load(study_dir)
    _reject_force_with_selection(args)
    names = select(models, args.variant)
    if args.jobs < 1:
        sys.exit(f"-j must be at least 1, got {args.jobs}")

    selection = _bins_per_variant(models, names, args)

    # Everything decidable up front is decided here, so a failure lands on
    # the terminal instead of in a log nobody is watching yet. The detached
    # supervisor is this same function with --foreground, so it re-runs all
    # of it -- these checks are idempotent by construction.
    if shutil.which("NumCalc") is None:
        sys.exit("NumCalc not found on PATH — install Mesh2HRTF's NumCalc "
                 "or add its directory to PATH.")

    try:
        options = solver_flags(nitermax=args.nitermax,
                               check_normals=args.check_normals,
                               adapt_fmmlength=args.adapt_fmmlength)
    except ValueError as e:
        sys.exit(str(e))

    busy = _busy(study_dir, names)
    if busy and not args.restart:
        # --restart kills these instead, below -- after detaching, so the
        # kill happens in the process that then does the work.
        where = "; ".join(f"{v} (pid {', '.join(map(str, pids))})"
                          for v, pids in sorted(busy.items()))
        sys.exit(f"NumCalc already running for: {where}\n"
                 f"  Wait for it, or re-run with --restart to kill and "
                 f"start over.")

    plan = {}
    digests = {}
    for v in names:
        wd = workdir(study_dir, v)
        nf = len(models[v].frequencies)
        present = set(available_bins(wd)) if wd.is_dir() else set()
        wanted = (set(range(1, nf + 1)) if selection[v] is None
                  else set(selection[v]))
        # Resume is the default: only bins without results are computed, so
        # an interrupted run continues and a complete one is a no-op.
        # --force keeps the whole selection and discards what is there.
        if not args.force:
            wanted -= present
        if not wanted:
            continue
        digests[v] = _model_digest(models[v])
        _guard_unchanged_model(wd, v, digests[v], present, args.force)
        plan[v] = sorted(wanted)

    if not plan:
        if _has_band_selection(args):
            n = sum(len(b or ()) for b in selection.values())
            print(f"  Nothing to run (all {n} selected bin(s) already "
                  f"present — drop the selector to compute the rest).")
        else:
            print("  Nothing to run (all bins present — use --force to "
                  "recompute).")
        return

    if not args.foreground:
        return _detach_solve(study_dir, args, plan)

    return _solve_foreground(study_dir, models, plan, digests, args, options)


def _busy(study_dir, names):
    """``{variant: pids}`` for variants with a live NumCalc.

    ``running_variants`` looks under ``<root>/numcalc/<variant>``, which is
    exactly ``runs/numcalc/<variant>`` when the root is the study's runs/
    directory — so the new layout needs no new scanner.
    """
    from pynumcalc.status import running_variants
    return running_variants(Path(study_dir) / RUNS_DIR, names)


def _solve_foreground(study_dir, models, plan, digests, args, options):
    from pynumcalc.packing import clear_pack
    from pynumcalc.status import solver_warnings

    if args.restart:
        busy = _busy(study_dir, list(plan))
        if busy:
            for v, pids in sorted(busy.items()):
                print(f"  {v}: killing NumCalc (pid {', '.join(map(str, pids))})")
            _kill_numcalc([p for pids in busy.values() for p in pids])
            still = _busy(study_dir, list(plan))
            if still:
                sys.exit(f"NumCalc survived --restart for: "
                         f"{', '.join(sorted(still))}")

    solver = solver_for()
    # One variant at a time: a Solver solves one Model, so -j bounds the
    # processes *within* a variant (the bin fan-out in pynumcalc.plan_jobs).
    # Peak memory -- the thing -j exists to bound -- is bounded either way;
    # what is lost against the old stage machine is running several small
    # variants side by side.
    print(f"\n{'=' * 60}\n  Running NumCalc "
          f"({len(plan)} variant(s), {args.jobs} process(es) at a time)"
          f"\n{'=' * 60}")

    failed = []
    for v, bins in plan.items():
        wd = workdir(study_dir, v, solver.name)
        wd.mkdir(parents=True, exist_ok=True)
        # A leftover be.npz shadows be.out/ everywhere (status, loaders), so
        # a run writing into be.out/ would be invisible behind stale packed
        # data. On a resume its bins are restored first so the runs merge.
        clear_pack(wd, restore=not args.force)
        if args.force:
            # NumCalc only wipes be.out/ when invoked without -istart/-iend,
            # so relying on it would leave stale bins behind whenever a
            # --force run fans out into ranges. Old logs go too, or their
            # warnings linger in `bem status`.
            shutil.rmtree(wd / "be.out", ignore_errors=True)
            for log in wd.glob("NC*.out"):
                log.unlink()
        # Written before the solve, not after: every bin NumCalc produces
        # from here on is described by this fingerprint, including the ones
        # a crash leaves behind.
        (wd / _DIGEST_FILE).write_text(digests[v] + "\n")

        nf = len(models[v].frequencies)
        # bins=None keeps NumCalc's plain invocation (and its NC.out log) for
        # a full fresh band; anything narrower states its range explicitly.
        scope = None if bins == list(range(1, nf + 1)) else bins
        print(f"  {v}: solving {len(bins)} bin(s) in {wd}")
        try:
            results = solver.solve(models[v], workdir=wd, bins=scope,
                                   jobs=args.jobs, options=options)
        except (RuntimeError, ValueError) as e:
            print(f"  {v}: FAILED — {e}")
            failed.append(v)
            continue
        # NumCalcSolver's ingest happens to write the same file today, so
        # this is belt-and-braces -- but "a finished solve leaves a
        # results.npz" is the CLI's promise, not a side effect it inherits.
        saved = results.save(wd / "results.npz")
        # Not "OK": run_jobs already prints that per job, and two OK lines
        # for one variant read as two things having happened.
        print(f"  {v}: results -> {saved}")

        issues = solver_warnings(wd)
        if issues:
            warned = sorted({b for _, b, _ in issues if b is not None})
            where = f" (bins {', '.join(map(str, warned))})" if warned else ""
            print(f"  {v}: {len(issues)} solver warning(s) in logs{where}"
                  f" — non-converged bins usually mean a defective mesh")

    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(failed)}")
    print("\nDone.")


def _kill_numcalc(pids, timeout=10.0):
    """Terminate NumCalc processes by PID, escalating to SIGKILL.

    Works on PIDs rather than Popen handles: a detached run's processes
    outlive the CLI invocation that started them, so there is no handle left.
    """
    import os
    import signal
    import time

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # someone else's process; not ours to wait on
            return False
        return True

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(alive(p) for p in pids):
            return
        time.sleep(0.1)
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _detach_solve(study_dir, args, plan):
    """Start the solve in a background copy of this CLI and return.

    NumCalc runs for hours, so the terminal should not be held for it. The
    supervisor is this same CLI re-invoked with --foreground: that preserves
    -j queueing, the resume arithmetic and the solver-warning summary
    exactly, rather than reimplementing them for the detached case. `bem
    status` already sees the jobs -- it derives them from a /proc scan, not
    from any bookkeeping this function would have to write.
    """
    argv = [sys.executable, "-m", "bemkit.cli", str(study_dir), "solve",
            "--foreground"]
    if args.variant:
        argv += ["--variant", *args.variant]
    if args.jobs:
        argv += ["-j", str(args.jobs)]
    if args.force:
        argv.append("--force")
    if args.restart:
        argv.append("--restart")
    if args.nitermax is not None:
        argv += ["--nitermax", str(args.nitermax)]
    if args.check_normals:
        argv.append("--check-normals")
    if args.adapt_fmmlength:
        argv.append("--adapt-fmmlength")
    # `is not None`, not truthiness: the supervisor re-resolves the selection
    # from these flags, so a selector that is present but empty must still be
    # forwarded as present — dropping it would silently run the whole band.
    if args.freq is not None:
        argv += ["--freq", *(str(f) for f in args.freq)]
    if args.bins is not None:
        argv += ["--bins", *(str(b) for b in args.bins)]
    if args.preview is not None:
        argv += ["--preview", str(args.preview)]

    log_path = Path(study_dir) / RUNS_DIR / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            argv, cwd=str(study_dir), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            # Its own session, or the run dies with the terminal that started
            # it (SIGHUP) or with a Ctrl-C aimed at this process group.
            start_new_session=True)

    n_bins = sum(len(b) for b in plan.values())
    print(f"  Started in background (pid {proc.pid}), {len(plan)} variant(s), "
          f"{n_bins} bin(s), {args.jobs} process(es) at a time")
    print(f"  Log: {log_path}")
    print(f"  Progress: bem {study_dir}")


# ── pack ───────────────────────────────────────────────────────────────────


def cmd_pack(study_dir, args):
    from pynumcalc.packing import has_results, pack_variant
    from pynumcalc.status import numcalc_pids

    models = load(study_dir)
    names = select(models, args.variant)
    solver = solver_for()

    packed = 0
    for v in names:
        wd = workdir(study_dir, v, solver.name)
        if not has_results(wd):
            print(f"  {v}: skipping (no results)")
            continue
        if numcalc_pids(wd):
            # Skip, not abort: packing deletes be.out/ from under a solver
            # still writing into it, but the finished variants should pack.
            print(f"  {v}: skipping (NumCalc still running)")
            continue
        pack_variant(wd, nf=len(models[v].frequencies),
                     delete_original=not args.keep_original)
        packed += 1
    if not packed:
        print("  Nothing packed.")


# ── estimate-ram ───────────────────────────────────────────────────────────


def cmd_estimate_ram(study_dir, args):
    """Report NumCalc's own RAM estimate for each variant.

    Runs the clustering pass only (``-estimate_ram``), no BEM solve, so it is
    fast. It answers the question our ``-j`` default of 1 currently answers
    by being conservative: how much memory one solve actually needs.

    Estimated at the **highest** frequency by default, because FMM memory
    grows with frequency and the top bin is the one that has to fit. NumCalc
    slightly *under*-estimates: the figure covers the FMM matrices, not the
    mesh and result arrays.

    The bundle is exported here rather than demanded of the user. There is no
    separate `export` command any more, so "solve first" would mean running
    the very solve this estimate exists to size — backwards. Exporting is
    cheap, deterministic, and writes nothing NumCalc's results depend on.
    """
    from pynumcalc.status import numcalc_pids

    if shutil.which("NumCalc") is None:
        sys.exit("NumCalc not found on PATH — install Mesh2HRTF's NumCalc "
                 "or add its directory to PATH.")

    models = load(study_dir)
    names = select(models, args.variant)
    solver = solver_for()

    for v in names:
        wd = workdir(study_dir, v, solver.name)
        nf = len(models[v].frequencies)
        top = float(models[v].frequencies[-1])
        if numcalc_pids(wd):
            # Rewriting NC.inp under a live solver is the one thing every
            # other command here refuses; no estimate is worth it.
            print(f"  {v}: skipping (NumCalc still running)")
            continue
        try:
            solver.export(models[v], wd)
        except ValueError as e:
            print(f"  {v}: cannot export — {e}")
            continue

        cmd = ["NumCalc", "-estimate_ram"]
        if not args.all_bins:
            cmd += ["-istart", str(nf), "-iend", str(nf)]
        scope = "whole band" if args.all_bins else f"bin {nf} ({top:.0f} Hz)"

        memory = wd / "Memory.txt"
        memory.unlink(missing_ok=True)
        proc = subprocess.run(cmd, cwd=str(wd), capture_output=True, text=True)
        if proc.returncode != 0 or not memory.exists():
            tail = (proc.stdout or proc.stderr).strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            print(f"  {v}: failed — {reason}")
            continue
        # Memory.txt rows are "<step> <frequency Hz> <GB>", no header.
        peak = 0.0
        for line in memory.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    peak = max(peak, float(parts[2]))
                except ValueError:
                    pass
        print(f"  {v}: {peak:.3f} GB  ({scope})")


# ── ps ─────────────────────────────────────────────────────────────────────


def cmd_ps(study_dir, args):
    """List every running NumCalc, across all studies. STUDY_DIR is ignored.

    The rest of the CLI is scoped to one study; once runs detach, the missing
    view is "which project is computing right now". Each live process reveals
    that in its working directory (``<study>/runs/<solver>/<variant>/``), so
    this reads the answer straight off a /proc scan.
    """
    from bemkit.cli.study import load_study
    from pynumcalc.status import available_bins, scan_numcalc

    jobs = scan_numcalc()
    if not jobs:
        print("No NumCalc processes running.")
        return

    nf_cache = {}
    rows = []
    for nc_dir, pids in jobs.items():
        variant = nc_dir.name
        # <study>/runs/<solver>/<variant>/ is the layout every bundle this
        # CLI writes uses; anything else was launched by hand, so fall back
        # to the raw path.
        study = (nc_dir.parents[2]
                 if len(nc_dir.parents) > 2 and nc_dir.parents[1].name == RUNS_DIR
                 else None)
        name = study.name if study is not None else str(nc_dir)

        nf = None
        if study is not None:
            if study not in nf_cache:
                try:
                    nf_cache[study] = {
                        v: len(m.frequencies)
                        for v, m in load_study(study).items()}
                except Exception:
                    # An unreadable or broken study.py must cost one column,
                    # not the whole listing — and model.py is arbitrary
                    # Python, so "broken" covers more than a missing file.
                    nf_cache[study] = {}
            nf = nf_cache[study].get(variant)

        done = len(available_bins(nc_dir))
        bins = f"{done}/{nf} bins" if nf else f"{done} bins"
        rows.append((name, variant, bins, len(pids), pids))

    rows.sort(key=lambda r: (r[0], r[1]))
    n_studies = len({r[0] for r in rows})
    n_proc = sum(r[3] for r in rows)
    print(f"NumCalc running ({n_studies} study/studies, {n_proc} process(es)):")

    mw = max(len(r[0]) for r in rows)
    vw = max(len(r[1]) for r in rows)
    for name, variant, bins, n, pids in rows:
        proc = f"{n} proc"
        pidstr = "pid " + ", ".join(map(str, pids))
        print(f"  {name:<{mw}}  {variant:<{vw}}  {bins:>10}  {proc:>7}  {pidstr}")
