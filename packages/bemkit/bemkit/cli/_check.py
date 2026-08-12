"""`bem check` — mesh and setup diagnostics, over the built Model.

NumCalc validates almost nothing and returns *some* result regardless, so a
flipped normal or an under-resolved band shows up as a plausible-looking
result rather than an error. These are the checks it does not make; see
``bemkit/geometry/quality.py`` for what each one tests and why.

Over a Model there is nothing to look up: the mesh, the band and the medium
are attributes, and there is no mesh file to be missing — ``build()`` either
produced a Model or raised.
"""

import sys

_SYMBOL = {"ok": "✓", "warning": "!", "error": "✗"}


def register(sub):
    p = sub.add_parser(
        "check", help="Mesh and setup diagnostics (orientation, resolution, "
                      "element quality, source clearance)")
    p.add_argument("--variant", nargs="+", metavar="NAME",
                   help="Only check variants whose name starts with NAME")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on warnings too, not only errors")
    p.set_defaults(func=cmd_check)


def cmd_check(study_dir, args):
    from bemkit.cli._commands import load, select
    from bemkit.geometry.quality import LEVELS, check_mesh, worst_level
    from bemkit.model.sources import PointSource

    models = load(study_dir)
    names = select(models, args.variant)

    overall = "ok"
    for variant in names:
        model = models[variant]
        mesh = model.mesh
        # Only point sources: Hint 3 compares a source *position* against the
        # average edge length, and a plane wave carries a direction of
        # travel, which as a coordinate is nonsense.
        sources = [s.position for s in model.sources
                   if isinstance(s, PointSource)]

        findings = check_mesh(
            mesh.vertices, mesh.triangles,
            max_frequency=float(model.frequencies[-1]),
            speed_of_sound=model.medium.c,
            sources=sources)
        print(f"\n{variant}  ({mesh.n_nodes} nodes, {mesh.n_elements} elements, "
              f"up to {model.frequencies[-1]:.0f} Hz)")
        for f in findings:
            print(f"  {_SYMBOL[f.level]} {f.check}: {f.message}")
        overall = max(overall, worst_level(findings), key=LEVELS.index)

    if overall == "error" or (args.strict and overall == "warning"):
        sys.exit(1)
