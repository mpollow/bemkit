"""Pipeline subcommands: mesh / export / run / pack / all, plus status."""

import sys


def register(sub):
    p = sub.add_parser("mesh", help="Generate mesh from geometry/*.step or mesh/*.ply")
    p.set_defaults(func=cmd_mesh)

    p = sub.add_parser("export", help="Fieldpoints + mesh + NC.inp (all variants)")
    _add_variant(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("run", help="Run NumCalc (parallel)")
    _add_variant(p)
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                   help="Max parallel NumCalc instances (default: all)")
    p.add_argument("--split", type=int, default=None, metavar="N",
                   help="Fan each variant out as N bin-range processes (-istart/-iend)")
    p.add_argument("--missing-only", action="store_true",
                   help="Only run frequency bins absent from be.out/ (resume)")
    p.add_argument("--force", action="store_true",
                   help="Allow a full re-run to discard already-complete results")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pack", help="Pack be.out/ into be.npz (~4x shrink)")
    _add_variant(p)
    p.add_argument("--delete-original", action="store_true",
                   help="Remove be.out/ after a verified round-trip")
    p.set_defaults(func=cmd_pack)

    p = sub.add_parser("all", help="mesh -> export -> run -> plot")
    _add_variant(p)
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N")
    p.add_argument("--split", type=int, default=None, metavar="N")
    p.add_argument("--missing-only", action="store_true")
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
        split=args.split, missing_only=args.missing_only, force=args.force)
    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(sorted({j.name for j in failed}))}")


def cmd_pack(model_dir, args):
    from bemkit.cli import load_model_config
    _run_stage("pack", load_model_config(model_dir), prefixes=args.variant,
               delete_original=args.delete_original)


def cmd_all(model_dir, args):
    from bemkit.cli import load_model_config
    from bemkit.cli._plot import run_default_plots
    config = load_model_config(model_dir)
    _run_stage("mesh", config)
    _run_stage("export", config, prefixes=args.variant)
    completed, failed = _run_stage(
        "run", config, prefixes=args.variant, max_jobs=args.jobs,
        split=args.split, missing_only=args.missing_only, force=args.force)
    if failed:
        sys.exit(f"NumCalc failed for: {', '.join(sorted({j.name for j in failed}))}")
    run_default_plots(model_dir, prefixes=args.variant)
    print("\nDone.")


def show_status(model_dir):
    """Default command: per-variant stage checklist + next-step hint."""
    from bemkit.cli import load_model_config
    from bemkit.pipeline import STAGES
    from bemkit.sim.status import available_bins, solver_warnings

    config = load_model_config(model_dir)
    variants = config.variants(None)
    if not variants:
        pre_scripts = sorted(model_dir.glob("pre_*.[ps][yh]"))
        if pre_scripts:
            print(f"\n{model_dir}  (no mesh files yet)\n")
            print(f"  Found pre_* scripts: {', '.join(s.name for s in pre_scripts)}")
            print(f"\n  Next: bem {model_dir} mesh")
        else:
            print(f"\n{model_dir}  (no mesh files and no pre_* scripts)\n")
        return

    print(f"\n{model_dir}  (variants: {', '.join(variants)})\n")
    nf = config.nbins - 1
    order = ["export", "run", "pack"]
    needs = {name: [] for name in order}
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
                issues = solver_warnings(nc_dir)
                if issues:
                    bins = sorted({b for _, b, _ in issues if b is not None})
                    where = f": bins {', '.join(map(str, bins))}" if bins else ""
                    extra += f"  ⚠ {len(issues)} solver warning(s){where}"
            print(f"    {name:<7} {'✓' if done else '✗'}{extra}")
            if not done:
                needs[name].append(v)
        plots = model_dir / "plots"
        has_plots = plots.is_dir() and any(plots.glob(f"{v}_*"))
        print(f"    {'plot':<7} {'✓' if has_plots else '✗'}")
        print()

    from bemkit.meshing.io import MESH_EXTENSIONS

    geom_dir = model_dir / "geometry"
    mesh_dir = model_dir / "mesh"
    if geom_dir.is_dir() and mesh_dir.is_dir():
        mesh_files = sorted(f for f in mesh_dir.iterdir()
                            if f.suffix in MESH_EXTENSIONS)
        mesh_stems = {f.stem.split('.')[0] for f in mesh_files}
        unmeshed = sorted(f.stem for f in geom_dir.iterdir()
                          if f.suffix == '.step' and f.stem not in mesh_stems)
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
        if needs[name]:
            n, total = len(needs[name]), len(variants)
            count = f"({n}/{total} variants)" if total > 1 else ""
            hint = " --missing-only" if name == "run" else ""
            print(f"  Next: bem {rel} {name}{hint}  {count}".rstrip())
            break
    else:
        print("  All stages complete.")
