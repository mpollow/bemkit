"""`bem MODEL check` — mesh and setup diagnostics, over a config.toml model.

Frozen copy of the pre-spine `_check.py`, kept for the legacy CLI's tests;
`_check.py` itself now runs over a built Model. Goes when config.py does.

NumCalc validates almost nothing and returns *some* result regardless, so a
flipped normal or an under-resolved band shows up as a plausible-looking plot
rather than an error. These are the checks it does not make; see
``bemkit/meshing/quality.py`` for what each one tests and why.
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


def cmd_check(model_dir, args):
    from bemkit.cli import load_model_config
    from bemkit.geometry.io import find_mesh_file, read_mesh_triangles
    from bemkit.geometry.quality import LEVELS, check_mesh, worst_level

    config = load_model_config(model_dir)
    variants = config.variants(args.variant)
    if not variants:
        sys.exit(f"No mesh files in {model_dir / 'mesh'}. Run 'mesh' first.")

    # Point sources are in meters, like the mesh once read — Hint 3 compares
    # them directly against the average edge length.
    sources = [row[1:4] for row in
               (config.raw.get("POINT SOURCES") or {}).values()
               if isinstance(row, list) and len(row) == 8]

    max_freq = float(config.freq_array()[-1])
    overall = "ok"
    for variant in variants:
        stem = config.mesh_map().get(variant, variant)
        try:
            mesh_file = find_mesh_file(model_dir / "mesh", stem)
        except FileNotFoundError:
            print(f"\n{variant}: no mesh yet — run 'bem {model_dir} mesh'")
            continue

        vertices, triangles = read_mesh_triangles(mesh_file)
        findings = check_mesh(vertices, triangles, max_frequency=max_freq,
                              speed_of_sound=config.speed_of_sound,
                              sources=sources)
        print(f"\n{variant}  ({mesh_file.name}: {len(vertices)} nodes, "
              f"{len(triangles)} elements)")
        for f in findings:
            print(f"  {_SYMBOL[f.level]} {f.check}: {f.message}")
        overall = max(overall, worst_level(findings), key=LEVELS.index)

    if overall == "error" or (args.strict and overall == "warning"):
        sys.exit(1)
