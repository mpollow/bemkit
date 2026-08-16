"""`bem STUDY convergence` — per-bin solver quality, as a table and a figure.

Thin on purpose: argument parsing, a text summary that needs nothing but
numcalckit, and a call into :mod:`bemkit.view.convergence` for the drawing.
"""


def register(sub):
    p = sub.add_parser(
        "convergence",
        help="CGS iterations, final residual and cost per frequency bin")
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", metavar="PREFIX", default=None,
                   action=VariantAction,
                   help="Variant prefix filter, repeatable")
    p.add_argument("--save", metavar="PATH", default=None,
                   help="Write the figure to PATH instead of opening a window")
    p.add_argument("--no-plot", action="store_true", dest="no_plot",
                   help="Print the summary table only — needs no matplotlib")
    p.add_argument("--nitermax", type=int, default=None,
                   help="Iteration ceiling the run used, if -nitermax was "
                        "passed to NumCalc (default: NumCalc's own 250)")
    p.set_defaults(func=cmd_convergence)


def _summarise(name, records):
    """One aligned line per variant, for the text summary."""
    import numpy as np

    head = f"  {name:34s}"
    if not records:
        return f"{head} no NC*.out logs"
    iters = [r.iterations for r in records if r.iterations is not None]
    errs = [r.rel_error for r in records if r.rel_error is not None]
    if not iters:
        return f"{head} {len(records):3d} bins, none solved yet"

    line = (f"{head} {len(records):3d} bins   "
            f"iters {int(np.median(iters)):4d} med {max(iters):4d} max")
    if errs:
        line += f"   worst residual {max(errs):.1e}"
    warned = [r.bin for r in records if r.warnings]
    if warned:
        line += f"   ⚠ did not converge: bins {', '.join(map(str, warned))}"
    return line


def cmd_convergence(study_dir, args):
    import sys

    from numcalckit.status import convergence
    from bemkit.cli._commands import load, select
    from bemkit.study import workdir

    models = load(study_dir)
    names = select(models, args.variant)

    # Deliberately not prepared_bundles(): that exports a bundle when one is
    # missing, and merely reading logs must never write into a study.
    records = {v: convergence(workdir(study_dir, v)) for v in names}

    print(f"\n{study_dir}  (convergence over {len(names)} variant(s))\n")
    for v in names:
        print(_summarise(v, records[v]))

    solved = {v: recs for v, recs in records.items()
              if any(r.iterations is not None for r in recs)}
    missing = [v for v in names if v not in solved]
    if missing:
        print(f"\n  {len(missing)} variant(s) with no solver log: "
              f"{', '.join(missing)}")
    if not solved:
        sys.exit(
            "\nNo NC*.out logs to read. They are not checked in, and "
            "`bem solve --force` deletes them — re-solve to regenerate.")

    warned = sorted(v for v, recs in solved.items()
                    if any(r.warnings for r in recs))
    if warned:
        print(f"\n  ⚠ non-converged: {', '.join(warned)}\n")
    else:
        print("\n  ✓ every solved bin reached NumCalc's tolerance\n")

    if args.no_plot:
        return 0

    from bemkit.cli import require
    require("matplotlib", "view")
    import matplotlib
    if args.save:
        # Chosen before pyplot is imported: picking a backend afterwards is
        # a no-op, and a headless `--save` must not need a display.
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from numcalckit.status import NITER_MAX_DEFAULT
    from bemkit.view.convergence import plot_convergence

    fig = plot_convergence(
        solved, title=f"{study_dir.name} — solver convergence",
        niter_max=args.nitermax or NITER_MAX_DEFAULT)
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"  wrote {args.save}")
    else:
        plt.show()
    return 0
