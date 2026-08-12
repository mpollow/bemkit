"""3D viewers: `bem STUDY geometry` and `bem STUDY results SET` (pyvista).

``geometry`` runs on the model.py spine: it loads the study, exports any
variant whose bundle is missing, and draws what NumCalc would be given.

``results`` does not, yet — see :func:`cmd_results`. Its parser is kept whole
so the flags stay documented and `bem STUDY results --help` still works; only
the command body refuses.
"""

import argparse


def register(sub):
    p = sub.add_parser("geometry", help="3D pre-run inspection: mesh + BC + eval grid")
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", default=None, action=VariantAction,
                   help="Variant prefix filter, repeatable (e.g., --variant sphcap-50-10)")
    p.add_argument("--no-mesh", action="store_true", help="Don't show ObjectMesh")
    p.add_argument("--no-grid", action="store_true", help="Don't show EvalGrid")
    p.add_argument("--mesh-opacity", type=float, default=1.0,
                   help="Mesh opacity (default: 1.0 — opaque. Lower it to see "
                        "eval points inside the object, at the cost of "
                        "superimposing the far side of the surface on the near one)")
    p.add_argument("--mesh-color", default="lightblue", help="Mesh color (default: lightblue)")
    p.add_argument("--grid-color", default="blue", help="Grid point color (default: blue)")
    p.add_argument("--point-size", type=float, default=6.0, help="Grid point size (default: 6.0)")
    # Edges are on by default now; kept so old invocations still parse.
    p.add_argument("--show-edges", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-edges", action="store_true",
                   help="Don't draw mesh edges on the simulated elements")
    p.add_argument("--no-bc", action="store_true", help="Don't color mesh by boundary condition")
    p.add_argument("--no-mirror", action="store_true",
                   help="Don't draw the mirror-plane ghost copies (the "
                        "SYMMETRY block of the deck; show only the meshed "
                        "elements)")
    p.add_argument("--bc-vibrating-color", default="red",
                   help="Color for vibrating elements (default: red), e.g., 'yellow', 'orange'")
    p.add_argument("--camera", default="iso", choices=["xz", "xy", "yz", "iso"],
                   help="Camera position (default: iso)")
    p.add_argument("--screenshot", help="Save screenshot to file instead of showing")
    p.add_argument("--background", default="white",
                   help="Background color (default: white). E.g. black, grey, #1a1a1a")
    p.set_defaults(func=cmd_geometry)

    p = sub.add_parser("results", help="3D post-run inspection: surface + field pressure")
    p.add_argument("set_name", nargs="?", default=None,
                   help="Fieldpoint set to show (e.g., 'onaxis'; "
                        "default: all sets)")
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", default=None, action=VariantAction,
                   help="Variant prefix filter, repeatable (e.g., --variant sphcap-50-10)")
    p.add_argument("--freq", type=float, default=1000.0,
                   help="Target frequency in Hz (default: 1000)")
    p.add_argument("--quantity", default="magnitude_db",
                   choices=["magnitude_db", "magnitude", "phase", "real", "imag"],
                   help="Quantity to display (default: magnitude_db)")
    p.add_argument("--surface-quantity", default=None,
                   choices=["magnitude_db", "magnitude", "phase", "real", "imag"],
                   help="Quantity for surface (default: same as --quantity)")
    p.add_argument("--points-quantity", default=None,
                   choices=["magnitude_db", "magnitude", "phase", "real", "imag"],
                   help="Quantity for eval points (default: same as --quantity)")
    p.add_argument("--no-surface", action="store_true",
                   help="Skip surface pressure coloring")
    p.add_argument("--no-points", action="store_true",
                   help="Skip eval grid points")
    p.add_argument("--no-mirror", action="store_true",
                   help="Don't draw the mirror-plane ghost copies (the "
                        "SYMMETRY block of the deck; show only the meshed "
                        "elements)")
    p.add_argument("--no-edges", action="store_true",
                   help="Don't draw mesh edges on the simulated elements")
    p.add_argument("--no-phase-fade", action="store_true",
                   help="Disable magnitude-based white fade for phase plots "
                        "(show flat hsv phase colors instead)")
    p.add_argument("--norm", default="acc", choices=["acc", "raw"],
                   help="Pressure normalization for eval points (default: acc)")
    p.add_argument("--mesh-opacity", type=float, default=1.0,
                   help="Surface mesh opacity (default: 1.0)")
    p.add_argument("--point-size", type=float, default=6.0,
                   help="Eval point size (default: 6.0)")
    p.add_argument("--camera", default="iso",
                   choices=["xz", "xy", "yz", "iso"],
                   help="Camera position preset (default: iso)")
    p.add_argument("--screenshot", default=None,
                   help="Save screenshot to this path instead of showing")
    p.add_argument("--intensity", action="store_true",
                   help="Overlay acoustic intensity arrows I=½Re(p·v*) "
                        "from vEvalGrid (requires completed NumCalc run)")
    p.add_argument("--intensity-scale", type=float, default=1.0,
                   help="Arrow length relative to 5%% of grid extent (default: 1.0)")
    p.add_argument("--intensity-length", choices=["db", "magnitude", "uniform"],
                   default="db",
                   help="Arrow length mode: db (normalized dB, default), "
                        "magnitude (proportional to I in W/m²), "
                        "uniform (all same length)")
    p.add_argument("--intensity-clim", type=float, nargs=2, metavar=("LO", "HI"),
                   default=None,
                   help="dB range for --intensity-length db (e.g. -60 0). "
                        "Default: [peak-60, peak] of current frequency bin.")
    p.add_argument("--impedance", action="store_true",
                   help="Show eval points colored by specific acoustic impedance "
                        "Z=p/v_r (normalized to Z₀=Z/(ρ₀c₀))")
    p.add_argument("--impedance-quantity", default="real",
                   choices=["real", "imag", "magnitude", "magnitude_db", "phase"],
                   help="Impedance quantity to display (default: real → Re(Z₀)→1 far field)")
    p.add_argument("--plane", action="store_true",
                   help="Render eval points as a Delaunay-triangulated surface "
                        "(best for plane-type fieldpoint sets)")
    p.add_argument("--background", default="white",
                   help="Background color (default: white). E.g. black, grey, #1a1a1a")
    p.set_defaults(func=cmd_results)


def _load_bc_elements(nc_dir):
    """Boundary-condition element indices, read from the bundle's NC.inp.

    NC.inp rather than the Model, deliberately: what is drawn should be what
    the solver was actually given, so a stale bundle looks stale instead of
    silently showing the model.py you have now.
    """
    from pynumcalc.readers import read_nc_inp_boundary_conditions

    nc_inp_file = nc_dir / "NC.inp"
    if nc_inp_file.exists():
        return read_nc_inp_boundary_conditions(nc_inp_file)
    return None


def _prepared_bundles(study_dir, prefixes):
    """``(names, workdir_for)`` for the selected variants of a study.

    Exports any variant whose bundle is missing. There is no separate `bem
    export` command on the model.py spine — `estimate-ram` already exports
    what it needs for the same reason — so demanding a prepared bundle here
    would mean "run the solve you are trying to inspect first". Exporting is
    cheap, deterministic, and writes nothing a later solve depends on; a
    variant with a live NumCalc is left exactly as it is.
    """
    import sys

    from bemkit.cli._commands import load, select, solver_for, workdir
    from pynumcalc.status import numcalc_pids

    models = load(study_dir)
    names = select(models, prefixes)
    solver = solver_for()

    ready = []
    for name in names:
        wd = workdir(study_dir, name, solver.name)
        if (wd / "ObjectMesh" / "Nodes.txt").exists() and (wd / "NC.inp").exists():
            ready.append(name)
            continue
        if numcalc_pids(wd):
            print(f"  {name}: skipping (NumCalc still running)")
            continue
        try:
            solver.export(models[name], wd)
        except ValueError as e:
            print(f"  {name}: cannot export — {e}")
            continue
        ready.append(name)

    if not ready:
        sys.exit(f"No inspectable variant in {study_dir} "
                 f"(tried: {', '.join(names)})")
    return ready, lambda v: workdir(study_dir, v, solver.name)


def cmd_geometry(study_dir, args):
    from bemkit.cli import require
    require("pyvista", "viz3d")

    from pynumcalc.readers import (
        read_object_mesh_nodes, read_object_mesh_elements, read_eval_grid_nodes,
        read_nc_inp_symmetry,
    )
    from bemkit.view._symmetry import (
        ghost_geometry, describe_symmetry, EDGE_STYLE,
    )
    from bemkit.view.scene import setup_scene

    if not study_dir.exists():
        print(f"Error: Study directory not found: {study_dir}")
        raise SystemExit(1)

    variants, numcalc_dir = _prepared_bundles(study_dir, args.variant)
    current_variant = variants[0]

    def add_actors(scene, variant):
        """Add mesh, ghost and grid actors for the given variant.

        Returns (mesh_actor, ghost_actor, grid_actor).
        """
        nc_dir = numcalc_dir(variant)
        mesh_actor = None
        ghost_actor = None
        grid_actor = None

        if not args.no_mesh and (nc_dir / "ObjectMesh" / "Nodes.txt").exists():
            nodes = read_object_mesh_nodes(nc_dir)
            elements = read_object_mesh_elements(nc_dir)
            show_bc = not args.no_bc
            bc_elements = _load_bc_elements(nc_dir) if show_bc else None
            bc_colors = ({"rigid": "gray", "vibrating": args.bc_vibrating_color}
                         if show_bc else None)
            edges = not args.no_edges
            mesh_actor = scene.add_object_mesh(
                nodes, elements,
                opacity=args.mesh_opacity,
                color=args.mesh_color,
                show_edges=edges,
                color_by_bc=show_bc,
                bc_elements=bc_elements,
                bc_colors=bc_colors,
                **(EDGE_STYLE if edges else {}),
            )

            # Ghosts: the copies NumCalc reflects internally. Their own actor,
            # so --no-mirror is a pure skip -- but drawn at the same opacity
            # as the real elements. What sets the two apart is the wireframe,
            # which only the simulated elements carry.
            ghost = (None if args.no_mirror
                     else ghost_geometry(nodes, elements,
                                         read_nc_inp_symmetry(nc_dir / "NC.inp")))
            if ghost is not None:
                ghost_bc = None
                if bc_elements is not None and "vibrating" in bc_elements:
                    # Map the BC through the source index, so a mirrored
                    # vibrating patch shows up in the same colour.
                    import numpy as np
                    ghost_bc = {"vibrating": np.flatnonzero(
                        np.isin(ghost.source, bc_elements["vibrating"]))}
                ghost_actor = scene.add_object_mesh(
                    ghost.nodes, ghost.elements,
                    opacity=args.mesh_opacity,
                    color=args.mesh_color,
                    show_edges=False,
                    color_by_bc=show_bc and ghost_bc is not None,
                    bc_elements=ghost_bc,
                    bc_colors=bc_colors,
                )

        if not args.no_grid and (nc_dir / "EvalGrid" / "Nodes.txt").exists():
            points = read_eval_grid_nodes(nc_dir)
            grid_actor = scene.add_eval_grid(
                points,
                color=args.grid_color,
                point_size=args.point_size
            )

        return mesh_actor, ghost_actor, grid_actor

    # Print info
    print(f"Visualizing: {study_dir}  variant={current_variant}")
    if len(variants) > 1:
        print(f"  Variants: {', '.join(variants)}")

    # Create scene
    scene = setup_scene(screenshot=args.screenshot)
    scene._plotter.set_background(args.background)

    # Add initial actors
    state = {"variant": current_variant, "mesh_actor": None,
             "ghost_actor": None, "grid_actor": None}
    (state["mesh_actor"], state["ghost_actor"],
     state["grid_actor"]) = add_actors(scene, current_variant)

    if state["mesh_actor"] is None and state["grid_actor"] is None:
        print("Error: Nothing to visualize (both mesh and grid disabled/missing)")
        raise SystemExit(1)

    show_bc = not args.no_bc
    if state["mesh_actor"] is not None:
        if show_bc:
            print(f"  + ObjectMesh (opacity={args.mesh_opacity}, colored by BC: rigid=gray, vibrating={args.bc_vibrating_color})")
        else:
            print(f"  + ObjectMesh (opacity={args.mesh_opacity}, color={args.mesh_color})")
    if state["ghost_actor"] is not None:
        nc_dir = numcalc_dir(current_variant)
        print(f"  + Symmetry ghosts: "
              f"{describe_symmetry(read_nc_inp_symmetry(nc_dir / 'NC.inp'))}")
    if state["grid_actor"] is not None:
        print(f"  + EvalGrid (color={args.grid_color}, point_size={args.point_size})")

    # Variant text slider (interactive mode, >1 variant)
    if not args.screenshot and len(variants) > 1:
        def update_variant(v):
            if v == state["variant"]:
                return
            # Remove old actors
            for key in ("mesh_actor", "ghost_actor", "grid_actor"):
                if state[key] is not None:
                    scene._plotter.remove_actor(state[key])
                    state[key] = None
            state["variant"] = v
            # Add new
            (state["mesh_actor"], state["ghost_actor"],
             state["grid_actor"]) = add_actors(scene, v)
            scene._plotter.render()
            print(f"  → variant: {v}")

        scene.add_variant_slider(variants, update_variant)

    # Set camera
    scene.set_camera(args.camera)

    # Screenshot hotkey
    if not args.screenshot:
        scene.add_screenshot_hotkey()

    scene.show_or_screenshot(args.screenshot)




#: What `results` needs from the results layer that the model.py spine does
#: not yet provide. Listed rather than summarised so the message names the
#: actual gap instead of "not supported".
_RESULTS_GAPS = (
    "acceleration-normalized pressure (--norm acc, which needs the vibrating "
    "area A_vib)",
    "single-bin reads without loading the whole band",
    "the eval-point/boundary geometry that went with a solve",
)


def cmd_results(study_dir, args):
    """Refuse, with the reason.

    The interactive viewer was built on ``SimulationData``/``BEMResult``,
    which the model.py spine deliberately dropped: ``Results`` is
    store-backed and solver-neutral, carries no mesh, no field-point
    coordinates and no acc normalization. Porting the viewer onto it is the
    bemkit/view/ task, not something to fake here — a viewer showing
    plausible-but-unnormalized numbers is worse than one that will not start.
    """
    import sys

    gaps = "\n".join(f"    - {g}" for g in _RESULTS_GAPS)
    sys.exit(
        f"bem STUDY results is not yet ported to Python studies (see Task 15: "
        f"fold bemviz into bemkit/view/).\n"
        f"  The 3D results viewer needs, and Results does not yet expose:\n"
        f"{gaps}\n"
        f"  Working today: `bem {study_dir} geometry` (mesh, boundary "
        f"conditions and eval grid),\n"
        f"  and the numbers themselves via "
        f"`bemkit.Results.load(\"{study_dir}/runs/numcalc/<variant>\")`.")
