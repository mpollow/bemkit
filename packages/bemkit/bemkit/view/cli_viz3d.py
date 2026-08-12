"""3D viewers: `bem STUDY geometry` and `bem STUDY results SET` (pyvista).

Both run on the model.py spine: they load the study, export any variant whose
bundle is missing, and draw what NumCalc was actually given. ``geometry``
shows the problem, ``results`` shows the answer — see
:mod:`bemkit.view.results_viewer` for the latter's machinery, and
:mod:`bemkit.view.bundle` for where its geometry and numbers come from.

Thin on purpose: this module is argument parsing and actor wiring. Anything
that can be checked without a window lives in ``bundle.py`` or
``_quantities.py``.
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
    p.add_argument("--norm", default="raw", choices=["acc", "raw"],
                   help="Pressure normalization (default: raw — NumCalc's own "
                        "output). 'acc' divides by j*rho0*omega*A_vib and "
                        "needs a velocity-driven model; a scattering model "
                        "(plane wave / point source) has no A_vib and is "
                        "refused rather than shown as raw under an acc label")
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


def cmd_geometry(study_dir, args):
    from bemkit.cli import require
    require("pyvista", "view")

    from pynumcalc.readers import (
        read_object_mesh_nodes, read_object_mesh_elements, read_eval_grid_nodes,
        read_nc_inp_symmetry,
    )
    from bemkit.view._symmetry import (
        ghost_geometry, describe_symmetry, EDGE_STYLE,
    )
    from bemkit.view.bundle import prepared_bundles
    from bemkit.view.scene import setup_scene

    if not study_dir.exists():
        print(f"Error: Study directory not found: {study_dir}")
        raise SystemExit(1)

    variants, _models, numcalc_dir = prepared_bundles(study_dir, args.variant)
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


def cmd_results(study_dir, args):
    """`bem STUDY results [SET]` — the interactive 3D results viewer."""
    import sys

    from bemkit.cli import require
    require("pyvista", "view")

    from bemkit.results.normalization import NoVelocityDrive
    from bemkit.view._quantities import PHASE_DB_RANGE
    from bemkit.view.bundle import prepared_bundles
    from bemkit.view.results_viewer import ResultsViewer
    from bemkit.view.scene import setup_scene

    if not study_dir.exists():
        sys.exit(f"Error: Study directory not found: {study_dir}")

    if args.no_surface and args.no_points and not (args.intensity
                                                   or args.impedance):
        sys.exit("Nothing to visualize: --no-surface and --no-points with "
                 "neither --intensity nor --impedance.")

    variants, models, workdir_for = prepared_bundles(
        study_dir, args.variant, require_results=True)

    print(f"Visualizing: {study_dir}  variant={variants[0]}")
    if len(variants) > 1:
        print(f"  Variants: {', '.join(variants)}")

    scene = setup_scene(screenshot=args.screenshot)
    scene._plotter.set_background(args.background)

    viewer = ResultsViewer(scene, workdir_for, models, args)
    try:
        viewer.load_variant(variants[0])
    except NoVelocityDrive as e:
        sys.exit(f"--norm acc: {e}")

    if args.set_name is not None and args.set_name not in viewer.bundle.set_names:
        sys.exit(f"No field-point set {args.set_name!r} in {variants[0]} "
                 f"(available: {', '.join(viewer.bundle.set_names) or 'none'})")
    if args.set_name is None and not args.no_points:
        print(f"  Field-point sets: "
              f"{', '.join(viewer.bundle.set_names) or 'none'} "
              f"(name one to filter)")

    bin_idx = viewer.initial_bin(args.freq)
    actual = float(viewer.bundle.frequencies[bin_idx])
    print(f"  Requested {args.freq:.1f} Hz → bin {bin_idx + 1} "
          f"({actual:.1f} Hz)")

    viewer.build_variant(variants[0], bin_idx)

    if (viewer.surface_actor is None and viewer.points_actor is None
            and viewer.impedance_actor is None
            and viewer.intensity_actor is None):
        sys.exit("Nothing to visualize (surface and points both "
                 "unavailable or disabled).")

    if viewer.surface_actor is not None:
        how = (f"phase+alpha (top {PHASE_DB_RANGE:.0f} dB)"
               if viewer.surface_rgba else viewer.surface_quantity)
        print(f"  + Surface pressure ({how} @ {actual:.1f} Hz, "
              f"norm={args.norm}, opacity={args.mesh_opacity})")
    if viewer.points_actor is not None:
        how = (f"phase+alpha (top {PHASE_DB_RANGE:.0f} dB)"
               if viewer.points_rgba else viewer.points_quantity)
        print(f"  + Eval points: set={args.set_name or 'all'}, {how}, "
              f"norm={args.norm}, point_size={args.point_size}")

    viewer.setup_interactivity(variants)
    scene.show_or_screenshot(args.screenshot)
