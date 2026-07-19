"""Pipeline subcommands: mesh / export / run / pack / all, plus status."""

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

    p = sub.add_parser("run", help="Run NumCalc (parallel; computes missing "
                                   "bins only — resumes interrupted runs)")
    _add_variant(p)
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                   help="Max parallel NumCalc instances (default: all)")
    p.add_argument("--split", type=int, default=None, metavar="N",
                   help="Fan each variant out as N bin-range processes "
                        "(-istart/-iend); on a resume, splits the missing bins")
    p.add_argument("--force", action="store_true",
                   help="Recompute all bins from scratch, discarding existing "
                        "results (needed after re-exporting a changed mesh/config)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pack", help="Pack be.out/ into be.npz (~4x shrink)")
    _add_variant(p)
    p.add_argument("--delete-original", action="store_true",
                   help="Remove be.out/ after a verified round-trip")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("all", help="mesh -> export -> run")
    _add_variant(p)
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N")
    p.add_argument("--split", type=int, default=None, metavar="N")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_all)


def _add_variant(p):
    p.add_argument("--variant", nargs="+", metavar="PREFIX", default=None,
                   help="Only process variants matching these prefixes")


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
    completed, failed = _run_stage(
        "run", config, prefixes=args.variant, max_jobs=args.jobs,
        split=args.split, force=args.force)
    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(sorted({j.name for j in failed}))}")


def cmd_pack(model_dir, args):
    from bemkit.cli import load_model_config
    _run_stage("pack", load_model_config(model_dir), prefixes=args.variant,
               delete_original=args.delete_original)


def cmd_all(model_dir, args):
    from bemkit.cli import load_model_config
    config = load_model_config(model_dir)
    _run_stage("mesh", config)
    _run_stage("export", config, prefixes=args.variant)
    completed, failed = _run_stage(
        "run", config, prefixes=args.variant, max_jobs=args.jobs,
        split=args.split, force=args.force)
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
    for v in variants:
        print(f"  {v}:")
        nc_dir = model_dir / "numcalc" / v
        for name in order:
            done = STAGES[name].check(config, v)
            extra = ""
            if name == "run":
                n_avail = len(available_bins(nc_dir))
                if 0 < n_avail < nf:
                    extra = f"  ({n_avail}/{nf} bins)"
                    done = False
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
            print(f"    {name:<7} {'✓' if done else '✗'}{extra}")
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
            break
    else:
        print("  All stages complete.")
