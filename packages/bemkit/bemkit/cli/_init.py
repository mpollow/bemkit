"""Model bootstrap: write a config.toml for a mesh that already exists.

`bem MODEL init [MESH ...]` turns a ready-to-use mesh file into a model
directory that exports and runs. Everything the template writes that is not a
fixed default is measured off the mesh:

- fieldpoint radii clear the bounding box, so validation keeps the points
  (``make_fields_onaxis`` measures radii from the *origin*, not the mesh
  centre — fixed radii would land inside a large or off-centre mesh);
- a ``[BOUNDARY]`` anchor is synthesised only when the mesh carries no
  ``vibrating`` physical group, because a mesh with neither exports as a fully
  rigid surface with no source, which NumCalc solves to zero without complaint.
"""

import shutil
import sys
from pathlib import Path

#: Fixed solver defaults for a bootstrapped model.
INIT_SR = 32000
INIT_NBINS = 65

#: Elements per wavelength below which the mesh is too coarse for INIT_SR.
_MIN_ELEMENTS_PER_WAVELENGTH = 6


def register(sub):
    p = sub.add_parser("init", help="Bootstrap config.toml from an existing "
                                    "mesh (then: export, run)")
    p.add_argument("mesh", nargs="*", metavar="MESH",
                   help="Mesh file(s) to copy into mesh/ "
                        "(default: use whatever is already in mesh/)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing config.toml")
    p.set_defaults(func=cmd_init)


def _num(x):
    """Format a float for TOML, never as a bare integer."""
    s = f"{float(x):.4g}"
    return s if ("e" in s or "." in s) else s + ".0"


def _collect_meshes(model_dir, given):
    """Return the mesh files to bootstrap from, copying them into mesh/."""
    from bemkit.geometry.io import MESH_EXTENSIONS

    mesh_dir = model_dir / "mesh"
    if given:
        sources = []
        for raw in given:
            src = Path(raw).expanduser().resolve()
            if not src.is_file():
                sys.exit(f"Mesh file not found: {raw}")
            if src.suffix not in MESH_EXTENSIONS:
                sys.exit(f"Not a mesh file: {raw} "
                         f"(expected one of {', '.join(MESH_EXTENSIONS)})")
            sources.append(src)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for src in sources:
            dst = mesh_dir / src.name
            if not dst.exists() or src != dst.resolve():
                shutil.copy2(src, dst)
                print(f"  Copied {src.name} → mesh/")
            out.append(dst)
        return sorted(set(out), key=lambda f: f.name)

    existing = sorted((f for f in (mesh_dir.iterdir() if mesh_dir.is_dir() else [])
                       if f.suffix in MESH_EXTENSIONS), key=lambda f: f.name)
    if existing:
        return existing

    # Fall back to loose mesh files sitting in the model dir itself.
    loose = sorted((f for f in (model_dir.iterdir() if model_dir.is_dir() else [])
                    if f.is_file() and f.suffix in MESH_EXTENSIONS),
                   key=lambda f: f.name)
    if loose:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for src in loose:
            dst = mesh_dir / src.name
            shutil.copy2(src, dst)
            print(f"  Copied {src.name} → mesh/")
            out.append(dst)
        return out

    sys.exit(f"No mesh files found in {mesh_dir} (or {model_dir}). "
             f"Pass one explicitly: bem {model_dir} init path/to/model.msh "
             f"(accepted: {', '.join(MESH_EXTENSIONS)})")


def _inspect(mesh_file):
    """Measure the numbers the template is built from. Coordinates in metres."""
    import numpy as np

    from bemkit.geometry.io import (
        element_areas, read_mesh_physical_groups, read_mesh_triangles,
        resolve_vibrating_group,
    )

    vertices, triangles = read_mesh_triangles(mesh_file)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if diag <= 0:
        sys.exit(f"Degenerate mesh (zero extent): {mesh_file}")

    areas = element_areas(vertices, triangles)
    # Equilateral-equivalent edge length, as in examples/plate/compare_meshes.py.
    mean_edge = float(np.sqrt(4 * float(areas.mean()) / np.sqrt(3)))

    centroids = vertices[triangles].mean(axis=1)
    # Anchor on a real element centroid: resolve_boundary can then neither
    # reject it as too far nor select zero elements within the radius.
    anchor = centroids[int(np.argmax(centroids[:, 2]))]

    groups = read_mesh_physical_groups(mesh_file)
    found_name, _ = resolve_vibrating_group(groups)
    return {
        "n_elements": len(triangles),
        "n_nodes": len(vertices),
        "lo": lo, "hi": hi, "diag": diag,
        "mean_edge": mean_edge,
        "anchor": anchor,
        "vibrating_name": found_name,
        "candidate": _pick_driven_group(groups, len(triangles)),
        "groups": {k: len(v) for k, v in groups.items()},
    }


def _pick_driven_group(groups, n_elements):
    """Guess which physical group is the driven one, or None.

    A group covering every element carries no BC information (meshio writes
    one such catch-all for files with no real tags), so it is never a
    candidate. Among the rest the driven patch is the small one — a source
    disc or piston is a fraction of the radiating body.
    """
    candidates = {k: len(v) for k, v in groups.items() if 0 < len(v) < n_elements}
    if not candidates:
        return None
    return min(candidates, key=lambda k: (candidates[k], k))


def _compose(modelname, info, variants):
    """Build the config.toml text."""
    zmax, diag = float(info["hi"][2]), info["diag"]
    radii = [zmax + f * diag for f in (0.5, 1.0, 2.0)]
    lines = [
        f'modelname = "{modelname}"',
        "",
        "# Bootstrapped by `bem init` from an existing mesh.",
        "# Files in mesh/ are trusted and never remeshed; each one becomes its",
        "# own variant (numcalc/<stem>/), so no [mesh] or [variants] section is",
        f"# needed here. Variants: {', '.join(variants)}",
        "",
        "[bem]",
        f"sr = {INIT_SR}                # max sim freq = sr/2 = "
        f"{INIT_SR // 2000} kHz",
        f"nbins = {INIT_NBINS}                # {INIT_NBINS - 1} bins, "
        f"{(INIT_SR // 2) // (INIT_NBINS - 1)} Hz spacing",
        "air_density = 1.2",
        "speed_of_sound = 340.0",
        "bem_type = 4              # 0=TBEM, 1=SLFMBEM, 4=DMLFMBEM [default]",
        "time_factor = -1.0",
        "is_internal = 0           # 1 for an interior problem",
        "",
        "# Evaluation points along +z, placed clear of the mesh bounding box",
        f"# (mesh spans z up to {_num(zmax)} m, diagonal {_num(diag)} m).",
        "[fieldpoints.onaxis]",
        'type = "onaxis"',
        'dim = "z"',
        f"radii = [{', '.join(_num(r) for r in radii)}]   # metres",
    ]

    if info["vibrating_name"]:
        lines += [
            "",
            f"# The mesh carries a '{info['vibrating_name']}' physical group, used",
            "# as the velocity boundary condition — no [BOUNDARY] section needed.",
        ]
    elif info["candidate"]:
        others = [f"{n} ({c} elements)" for n, c in sorted(info["groups"].items())
                  if n != info["candidate"]]
        lines += [
            "",
            "# !! CHECK THIS !!  This mesh names its physical groups differently",
            "# from bemkit's own ('vibrating'), so the driven group is named here.",
            f"# `bem init` picked the smallest group, {info['candidate']} "
            f"({info['groups'][info['candidate']]} elements), which is usually",
            "# the source patch. Every element outside it is rigid.",
            f"# Other group(s) in this mesh: {', '.join(others) or 'none'}",
            "[groups]",
            f'vibrating = "{info["candidate"]}"',
        ]
    else:
        a = info["anchor"]
        lines += [
            "",
            "# !! CHECK THIS !!  The mesh has no 'vibrating' physical group, so",
            "# without a boundary condition the surface would be fully rigid and",
            "# the simulation would return silence. `bem init` therefore anchored",
            "# a driven patch on the topmost (+z) element of the mesh — a",
            "# placeholder, almost certainly not where your source belongs.",
            "# 'near' is a point in metres; every element centroid within",
            "# 'radius' vibrates. Both are resolved against the mesh at export,",
            "# so they survive a remesh.",
            "[BOUNDARY]",
            f"piston = {{ near = [{_num(a[0])}, {_num(a[1])}, {_num(a[2])}], "
            f"radius = {_num(0.05 * diag)}, velocity = 1.0 }}",
        ]
    return "\n".join(lines) + "\n"


def cmd_init(model_dir, args):
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = model_dir / "config.toml"
    if cfg_path.exists() and not args.force:
        sys.exit(f"{cfg_path} already exists — pass --force to overwrite it.")

    meshes = _collect_meshes(model_dir, args.mesh)
    primary = meshes[0]
    info = _inspect(primary)
    variants = [m.stem for m in meshes]

    cfg_path.write_text(_compose(model_dir.name, info, variants))

    print(f"\nWrote {cfg_path}")
    print(f"  Mesh:     {primary.name} "
          f"({info['n_elements']} elements, {info['n_nodes']} nodes)")
    print(f"  Extent:   {_num(info['diag'])} m diagonal, "
          f"mean edge {_num(info['mean_edge'] * 1000)} mm")
    print(f"  Variants: {', '.join(variants)}")
    groups = ", ".join(f"{n} ({c})" for n, c in sorted(info["groups"].items()))
    if info["groups"]:
        print(f"  Groups:   {groups}")
    if info["vibrating_name"]:
        print(f"  Source:   '{info['vibrating_name']}' physical group "
              f"({info['groups'][info['vibrating_name']]} elements)")
    elif info["candidate"]:
        print(f"  Source:   [groups] vibrating = \"{info['candidate']}\" "
              f"({info['groups'][info['candidate']]} elements, smallest group)")
        print("            ⚠ confirm that is the driven patch — if not, edit "
              "[groups] in config.toml")
    else:
        print("  Source:   no usable physical groups — wrote a placeholder "
              "[BOUNDARY] anchor")
        print("            ⚠ edit the 'piston' anchor before running, or the "
              "result is meaningless")

    lam = 340.0 / (INIT_SR / 2)
    if info["mean_edge"] > lam / _MIN_ELEMENTS_PER_WAVELENGTH:
        per_wl = lam / info["mean_edge"]
        print(f"\n  ⚠ Mesh is coarse for sr = {INIT_SR}: mean edge "
              f"{_num(info['mean_edge'] * 1000)} mm is only {per_wl:.1f} "
              f"elements per wavelength at {INIT_SR // 2000} kHz "
              f"(want >= {_MIN_ELEMENTS_PER_WAVELENGTH}).")
        print("    Lower sr in config.toml, or use a finer mesh.")

    rel = (model_dir.relative_to(Path.cwd())
           if model_dir.is_relative_to(Path.cwd()) else model_dir)
    print(f"\n  Next: bem {rel} export   (then: bem {rel} run)")
