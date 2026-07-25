"""Pipeline subcommands: mesh / export / run / pack / all, plus status."""

import subprocess
import sys


def register(sub):
    p = sub.add_parser("status", help="Per-variant stage checklist incl. "
                                      "running NumCalc jobs (the default)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("mesh", help="Generate mesh from geometry/ sources "
                                    "(.step CAD, .ply/.stl/.obj scans)")
    p.set_defaults(func=cmd_mesh)

    p = sub.add_parser("export", help="Fieldpoints + mesh + NC.inp (all variants)")
    _add_variant(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("run", help="Run NumCalc (computes missing bins only "
                                   "— resumes interrupted runs)")
    _add_variant(p)
    _add_band(p)
    p.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                   help="Max NumCalc processes at once (default: 1). "
                        "Variants and bin ranges are fanned out "
                        "automatically to fill N processes")
    p.add_argument("--force", action="store_true",
                   help="Recompute all bins from scratch, discarding existing "
                        "results (needed after re-exporting a changed mesh/config)")
    p.add_argument("--foreground", action="store_true",
                   help="Wait for NumCalc instead of detaching (default is to "
                        "start in the background and return)")
    p.add_argument("--restart", action="store_true",
                   help="Kill NumCalc processes already running for the "
                        "selected variants, then run. Resumes missing bins; "
                        "add --force to recompute everything")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pack", help="Pack be.out/ into be.npz (~4x shrink) "
                                    "and remove be.out/")
    _add_variant(p)
    p.add_argument("--keep-original", action="store_true",
                   help="Keep be.out/ instead of removing it after the "
                        "verified round-trip")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("ps", help="List NumCalc processes running anywhere "
                                  "(model / variant / bins / pid). Ignores MODEL_DIR")
    p.set_defaults(func=cmd_ps)

    p = sub.add_parser("all", help="mesh -> export -> run")
    _add_variant(p)
    _add_band(p)
    p.add_argument("-j", "--jobs", type=int, default=1, metavar="N")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_all)


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


def _has_band_selection(args):
    """True if --freq/--bins/--preview was given (any one of them).

    The one place this is asked, spelled the same way everywhere it is
    asked: an empty list (unreachable via argparse's ``nargs="+"``, but not
    via a direct caller) must count as "given" here exactly as it does in
    ``_selected_bins``/``_reject_force_with_selection``/``_detach_run``'s
    argv forwarding — a truthiness check on the list would disagree.
    """
    return args.freq is not None or args.bins is not None or args.preview is not None


def _selected_bins(config, args):
    """Resolve --freq/--bins/--preview to bin indices, or None for the full band.

    Prints what was resolved, with the actual bin frequencies, because a
    requested frequency is snapped to the grid rather than honoured exactly.
    """
    from bemkit.config import resolve_bins

    if not _has_band_selection(args):
        return None

    try:
        bins = resolve_bins(config.sr, config.nbins, freq=args.freq,
                            bins=args.bins, preview=args.preview)
    except ValueError as e:
        sys.exit(str(e))

    freqs = config.freq_array()
    nf = config.nbins - 1
    print(f"  Partial band: {len(bins)} of {nf} bins")
    listed = ", ".join(f"{freqs[b - 1]:.1f} Hz (bin {b})" for b in bins)
    print(f"    {listed}")
    if args.freq is not None and len(bins) < len(set(args.freq)):
        print(f"    ({len(set(args.freq))} frequencies requested, "
              f"{len(bins)} distinct bins)")
    return bins


def _reject_force_with_selection(args):
    """--force and a bin selection cannot be combined.

    --force discards every existing bin because bin presence says nothing
    about which export produced it. Scoping it to a subset would leave the
    unnamed bins in place — results from two different exports in one
    be.out/, which is the corruption --force exists to prevent.
    """
    if not args.force:
        return
    if not _has_band_selection(args):
        return
    sys.exit(
        "--force discards every existing bin and cannot be combined with "
        "--freq/--bins/--preview: the bins you did not name would still "
        "come from the previous export.\n"
        "  Full recompute:  bem MODEL run --force\n"
        "  Fresh partial:   rm -r numcalc/<variant>/be.out  &&  "
        "bem MODEL run --preview 8")


def _run_stage(name, config, **kwargs):
    from bemkit.pipeline import STAGES
    try:
        return STAGES[name].run(config, **kwargs)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(str(e))


def cmd_status(model_dir, args):
    show_status(model_dir)


def cmd_mesh(model_dir, args):
    from bemkit.cli import load_model_config
    _run_stage("mesh", load_model_config(model_dir))


def cmd_export(model_dir, args):
    from bemkit.cli import load_model_config
    _run_stage("export", load_model_config(model_dir), prefixes=args.variant)


def cmd_run(model_dir, args):
    from bemkit.cli import load_model_config
    config = load_model_config(model_dir)
    _reject_force_with_selection(args)
    bins = _selected_bins(config, args)
    if not args.foreground:
        return _detach_run(model_dir, config, args, bins)
    completed, failed = _run_stage(
        "run", config, prefixes=args.variant, max_jobs=args.jobs,
        force=args.force, restart=args.restart, bins=bins)
    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(sorted({j.name for j in failed}))}")


def _detach_run(model_dir, config, args, bins=None):
    """Start the run in a background copy of this CLI and return.

    NumCalc runs for hours, so the terminal should not be held for it. The
    supervisor is this same CLI re-invoked with --foreground: that preserves
    -j queueing, the bin fan-out and the solver-warning summary exactly,
    rather than reimplementing them for the detached case. `bem status`
    already sees the jobs -- it derives them from a /proc scan, not from any
    bookkeeping this function would have to write.

    Everything that can be decided up front is decided here, so failures land
    on the terminal instead of in a log nobody is watching yet.
    """
    from bemkit.pipeline import busy_error, plan_run, require_numcalc
    from bemkit.sim.status import running_variants

    try:
        require_numcalc()
    except RuntimeError as e:
        sys.exit(str(e))

    variants = config.variants(args.variant)
    if not variants:
        sys.exit(f"No mesh files in {model_dir / 'mesh'}. Run 'export' first.")
    if not args.restart:
        # --restart kills these instead; the stage does it, after detaching.
        msg = busy_error(running_variants(model_dir, variants))
        if msg:
            sys.exit(msg)

    _, jobs = plan_run(config, variants, max_jobs=args.jobs, force=args.force,
                       bins=bins)
    if not jobs:
        if bins is not None:
            print(f"  Nothing to run (all {len(bins)} selected bin(s) already "
                  f"present — drop the selector to compute the rest).")
        else:
            print("  Nothing to run (all bins present — use --force to recompute).")
        return

    argv = [sys.executable, "-m", "bemkit.cli", str(model_dir), "run",
            "--foreground"]
    if args.variant:
        argv += ["--variant", *args.variant]
    if args.jobs:
        argv += ["-j", str(args.jobs)]
    if args.force:
        argv.append("--force")
    if args.restart:
        argv.append("--restart")
    # `is not None`, not truthiness: the supervisor re-resolves the selection
    # from these flags, so a selector that is present but empty must still be
    # forwarded as present — dropping it would silently run the whole band.
    if args.freq is not None:
        argv += ["--freq", *(str(f) for f in args.freq)]
    if args.bins is not None:
        argv += ["--bins", *(str(b) for b in args.bins)]
    if args.preview is not None:
        argv += ["--preview", str(args.preview)]

    log_path = model_dir / "numcalc" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            argv, cwd=str(model_dir), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            # Its own session, or the run dies with the terminal that started
            # it (SIGHUP) or with a Ctrl-C aimed at this process group.
            start_new_session=True)

    n_at_once = min(args.jobs, len(jobs))
    print(f"  Started in background (pid {proc.pid}), {len(jobs)} job(s), "
          f"{n_at_once} at a time")
    print(f"  Log: {log_path}")
    print(f"  Progress: bem {model_dir}")


def cmd_pack(model_dir, args):
    from bemkit.cli import load_model_config
    _run_stage("pack", load_model_config(model_dir), prefixes=args.variant,
               delete_original=not args.keep_original)


def cmd_ps(model_dir, args):
    """List every running NumCalc, across all models. MODEL_DIR is ignored.

    The rest of the CLI is scoped to one model; once runs detach, the missing
    view is "which project is computing right now". Each live process reveals
    that in its working directory (``<model>/numcalc/<variant>/``), so this
    reads the answer straight off a /proc scan.
    """
    from bemkit.sim.status import scan_numcalc, available_bins

    jobs = scan_numcalc()
    if not jobs:
        print("No NumCalc processes running.")
        return

    nf_cache = {}
    rows = []
    for nc_dir, pids in jobs.items():
        variant = nc_dir.name
        # <model>/numcalc/<variant>/ is the layout every exported bundle uses;
        # anything else was launched by hand, so fall back to the raw path.
        model_dir_of = (nc_dir.parent.parent
                        if nc_dir.parent.name == "numcalc" else None)
        model = model_dir_of.name if model_dir_of else str(nc_dir)

        nf = None
        if model_dir_of is not None:
            if model_dir_of not in nf_cache:
                try:
                    from bemkit.config import ModelConfig
                    nf_cache[model_dir_of] = ModelConfig.from_dir(model_dir_of).nbins - 1
                except Exception:  # unreadable/incomplete config — show bare count
                    nf_cache[model_dir_of] = None
            nf = nf_cache[model_dir_of]

        done = len(available_bins(nc_dir))
        bins = f"{done}/{nf} bins" if nf else f"{done} bins"
        rows.append((model, variant, bins, len(pids), pids))

    rows.sort(key=lambda r: (r[0], r[1]))
    n_models = len({r[0] for r in rows})
    n_proc = sum(r[3] for r in rows)
    print(f"NumCalc running ({n_models} model(s), {n_proc} process(es)):")

    mw = max(len(r[0]) for r in rows)
    vw = max(len(r[1]) for r in rows)
    for model, variant, bins, n, pids in rows:
        proc = f"{n} proc"
        pidstr = "pid " + ", ".join(map(str, pids))
        print(f"  {model:<{mw}}  {variant:<{vw}}  {bins:>10}  {proc:>7}  {pidstr}")


def cmd_all(model_dir, args):
    from bemkit.cli import load_model_config
    config = load_model_config(model_dir)
    _reject_force_with_selection(args)
    bins = _selected_bins(config, args)
    _run_stage("mesh", config)
    _run_stage("export", config, prefixes=args.variant)
    completed, failed = _run_stage(
        "run", config, prefixes=args.variant, max_jobs=args.jobs,
        force=args.force, bins=bins)
    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(sorted({j.name for j in failed}))}")
    print("\nDone.")


def show_status(model_dir):
    """Default command: per-variant stage checklist + next-step hint."""
    from bemkit.cli import load_model_config
    from bemkit.pipeline import STAGES
    from bemkit.sim.status import (
        available_bins, current_step, numcalc_pids, solver_warnings)

    config = load_model_config(model_dir)
    variants = config.variants(None)
    if not variants:
        from bemkit.meshing.generate import SURFACE_EXTENSIONS
        geom_dir = model_dir / "geometry"
        has_sources = geom_dir.is_dir() and any(
            f.suffix in ['.step'] + SURFACE_EXTENSIONS for f in geom_dir.iterdir())
        if has_sources or config.geometry_cfg.get("exec"):
            print(f"\n{model_dir}  (no mesh files yet)\n")
            print(f"  Next: bem {model_dir} mesh")
        else:
            print(f"\n{model_dir}  (no mesh files and no geometry sources)\n")
        return

    print(f"\n{model_dir}  (variants: {', '.join(variants)})\n")
    nf = config.nbins - 1
    order = ["mesh", "export", "run", "pack"]
    needs = {name: [] for name in order}
    running = []
    partial_runs = []
    for v in variants:
        print(f"  {v}:")
        nc_dir = model_dir / "numcalc" / v
        for name in order:
            done = STAGES[name].check(config, v)
            mark = "✓" if done else "✗"
            extra = ""
            if name == "run":
                n_avail = len(available_bins(nc_dir))
                if 0 < n_avail < nf:
                    extra = f"  ({n_avail}/{nf} bins)"
                    done = False
                    mark = "◐"
                    partial_runs.append(v)
                pids = numcalc_pids(nc_dir)
                if pids:
                    running.append(v)
                    step = current_step(nc_dir)
                    at = f", computing bin {step}" if step else ""
                    extra += (f"  ⟳ running ({len(pids)} NumCalc "
                              f"process(es){at})")
                issues = solver_warnings(nc_dir)
                if issues:
                    bins = sorted({b for _, b, _ in issues if b is not None})
                    where = f": bins {', '.join(map(str, bins))}" if bins else ""
                    extra += f"  ⚠ {len(issues)} solver warning(s){where}"
            print(f"    {name:<7} {mark}{extra}")
            if not done:
                needs[name].append(v)
        print()

    from bemkit.meshing.io import MESH_EXTENSIONS

    geom_dir = model_dir / "geometry"
    mesh_dir = model_dir / "mesh"
    # With [variants], meshes are named per variant, not per source — the
    # source/mesh stem comparison below would be meaningless.
    if geom_dir.is_dir() and mesh_dir.is_dir() and not config.raw.get("variants"):
        mesh_files = sorted(f for f in mesh_dir.iterdir()
                            if f.suffix in MESH_EXTENSIONS)
        from bemkit.meshing.generate import SURFACE_EXTENSIONS
        unmeshed = sorted(g.stem for g in geom_dir.iterdir()
                          if g.suffix in ['.step'] + SURFACE_EXTENSIONS
                          and not any(f.stem == g.stem for f in mesh_files))
        if unmeshed:
            print(f"  Geometries without meshes: {', '.join(unmeshed)}\n")

    numcalc_d = model_dir / "numcalc"
    if numcalc_d.is_dir():
        variant_set = set(variants)
        stale = sorted(d.name for d in numcalc_d.iterdir()
                       if d.is_dir() and d.name not in variant_set
                       and (d / "NC.inp").exists())
        if stale:
            print(f"  Stale numcalc dirs (no matching mesh): {', '.join(stale)}\n")

    from pathlib import Path
    rel = (model_dir.relative_to(Path.cwd())
           if model_dir.is_relative_to(Path.cwd()) else model_dir)
    for name in order:
        pending = needs[name]
        if name == "run" and pending:
            # Don't suggest `bem run` for variants NumCalc is working on —
            # and if every pending run is in flight, waiting IS the next step.
            pending = [v for v in pending if v not in running]
            if not pending:
                print(f"  NumCalc running ({', '.join(running)}) — "
                      f"re-run `bem` to refresh.")
                break
        if pending:
            n, total = len(pending), len(variants)
            count = f"({n}/{total} variants)" if total > 1 else ""
            print(f"  Next: bem {rel} {name}  {count}".rstrip())
            if name == "run" and partial_runs:
                # A partial run's bins are usable right now, whatever the
                # rest of the band still needs — but only when `run` is
                # itself the blocking stage, so this offers a continuation
                # of that exact advice rather than dangling off an unrelated
                # one (a different variant's mesh/export, or the "NumCalc
                # running" line just above, which is not an imperative).
                print(f"     or bem {rel} plot freq"
                      f"    # the bins already computed are usable now")
            break
    else:
        print("  All stages complete.")
