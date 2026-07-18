"""3D viewers: `bem MODEL geometry` and `bem MODEL results SET` (pyvista)."""


def register(sub):
    p = sub.add_parser("geometry", help="3D pre-run inspection: mesh + BC + eval grid")
    p.add_argument("--variant", action="append", default=None,
                   help="Variant prefix filter, repeatable (e.g., --variant sphcap-50-10)")
    p.add_argument("--no-mesh", action="store_true", help="Don't show ObjectMesh")
    p.add_argument("--no-grid", action="store_true", help="Don't show EvalGrid")
    p.add_argument("--mesh-opacity", type=float, default=0.6, help="Mesh opacity (default: 0.6)")
    p.add_argument("--mesh-color", default="lightblue", help="Mesh color (default: lightblue)")
    p.add_argument("--grid-color", default="blue", help="Grid point color (default: blue)")
    p.add_argument("--point-size", type=float, default=6.0, help="Grid point size (default: 6.0)")
    p.add_argument("--show-edges", action="store_true", help="Show mesh edges")
    p.add_argument("--no-bc", action="store_true", help="Don't color mesh by boundary condition")
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
    p.add_argument("--variant", action="append", default=None,
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


def _load_bc_elements(nc_dir, model_dir):
    """Load boundary condition elements from NC.inp or .msh fallback."""
    from bemkit.meshing.io import read_mesh_physical_groups
    from bemkit.sim.readers import read_nc_inp_boundary_conditions

    nc_inp_file = nc_dir / "NC.inp"
    if nc_inp_file.exists():
        return read_nc_inp_boundary_conditions(nc_inp_file)

    # Fall back to .msh physical groups
    mesh_search_dir = model_dir / "mesh"
    msh_files = list(mesh_search_dir.glob("*.msh")) if mesh_search_dir.exists() else []
    if not msh_files:
        msh_files = list(model_dir.glob("*.msh"))
    if msh_files:
        return read_mesh_physical_groups(msh_files[0])
    return None


def cmd_geometry(model_dir, args):
    from bemkit.cli import require
    require("pyvista", "viz3d")

    from bemkit.sim.discovery import discover_variants, numcalc_dir
    from bemkit.sim.readers import (
        read_object_mesh_nodes, read_object_mesh_elements, read_eval_grid_nodes,
    )
    from bemkit.viz.scene import setup_scene

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        raise SystemExit(1)

    # Discover variants
    variants = discover_variants(model_dir, args.variant, need_object_mesh=True)

    if not variants:
        # Fallback: legacy single-variant layout (numcalc/ObjectMesh/ directly)
        if (model_dir / "numcalc" / "ObjectMesh" / "Nodes.txt").exists():
            variants = [None]  # None signals legacy layout
        else:
            if args.variant:
                print(f"Error: No prepared variants matching: {', '.join(args.variant)}")
            else:
                print(f"Error: No prepared variants found in {model_dir / 'numcalc'}")
            raise SystemExit(1)

    current_variant = variants[0]

    def add_actors(scene, variant):
        """Add mesh and grid actors for the given variant. Returns (mesh_actor, grid_actor)."""
        nc_dir = numcalc_dir(model_dir, variant)
        mesh_actor = None
        grid_actor = None

        if not args.no_mesh and (nc_dir / "ObjectMesh" / "Nodes.txt").exists():
            nodes = read_object_mesh_nodes(nc_dir)
            elements = read_object_mesh_elements(nc_dir)
            show_bc = not args.no_bc
            bc_elements = _load_bc_elements(nc_dir, model_dir) if show_bc else None
            mesh_actor = scene.add_object_mesh(
                nodes, elements,
                opacity=args.mesh_opacity,
                color=args.mesh_color,
                show_edges=args.show_edges,
                color_by_bc=show_bc,
                bc_elements=bc_elements,
                bc_colors={"rigid": "gray", "vibrating": args.bc_vibrating_color} if show_bc else None
            )

        if not args.no_grid and (nc_dir / "EvalGrid" / "Nodes.txt").exists():
            points = read_eval_grid_nodes(nc_dir)
            grid_actor = scene.add_eval_grid(
                points,
                color=args.grid_color,
                point_size=args.point_size
            )

        return mesh_actor, grid_actor

    # Print info
    if current_variant is not None:
        print(f"Visualizing: {model_dir}  variant={current_variant}")
        if len(variants) > 1:
            print(f"  Variants: {', '.join(variants)}")
    else:
        print(f"Visualizing: {model_dir}")

    # Create scene
    scene = setup_scene(screenshot=args.screenshot)
    scene._plotter.set_background(args.background)

    # Add initial actors
    state = {"variant": current_variant, "mesh_actor": None, "grid_actor": None}
    state["mesh_actor"], state["grid_actor"] = add_actors(scene, current_variant)

    if state["mesh_actor"] is None and state["grid_actor"] is None:
        print("Error: Nothing to visualize (both mesh and grid disabled/missing)")
        raise SystemExit(1)

    show_bc = not args.no_bc
    if state["mesh_actor"] is not None:
        if show_bc:
            print(f"  + ObjectMesh (opacity={args.mesh_opacity}, colored by BC: rigid=gray, vibrating={args.bc_vibrating_color})")
        else:
            print(f"  + ObjectMesh (opacity={args.mesh_opacity}, color={args.mesh_color})")
    if state["grid_actor"] is not None:
        print(f"  + EvalGrid (color={args.grid_color}, point_size={args.point_size})")

    # Variant text slider (interactive mode, >1 variant)
    if not args.screenshot and len(variants) > 1 and variants[0] is not None:
        def update_variant(v):
            if v == state["variant"]:
                return
            # Remove old actors
            for key in ("mesh_actor", "grid_actor"):
                if state[key] is not None:
                    scene._plotter.remove_actor(state[key])
                    state[key] = None
            state["variant"] = v
            # Add new
            state["mesh_actor"], state["grid_actor"] = add_actors(scene, v)
            scene._plotter.render()
            print(f"  → variant: {v}")

        scene.add_variant_slider(variants, update_variant)

    # Set camera
    scene.set_camera(args.camera)

    # Screenshot hotkey
    if not args.screenshot:
        scene.add_screenshot_hotkey()

    scene.show_or_screenshot(args.screenshot)


def cmd_results(model_dir, args):
    from bemkit.cli import require
    require("pyvista", "viz3d")
    require("matplotlib", "viz3d")

    from bemkit.config import ModelConfig, freq_bin
    from bemkit.sim.discovery import discover_variants
    from bemkit.viz.results_viewer import ResultsViewer, PHASE_DB_RANGE
    from bemkit.viz.scene import setup_scene

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        raise SystemExit(1)

    # Load config
    vars_file = model_dir / "config.toml"
    if not vars_file.exists():
        print(f"Error: config.toml not found in {model_dir}")
        raise SystemExit(1)
    config = ModelConfig.from_dir(model_dir)

    modelname = config["modelname"]
    freqs = config.freq_array()
    bin_idx, _, actual_freq = freq_bin(args.freq, config["sr"], config["nbins"])
    print(f"Model: {modelname}")
    print(f"Requested freq: {args.freq:.1f} Hz → nearest bin: {actual_freq:.1f} Hz")

    # Determine what to show (set_name=None means all fieldpoint sets)
    show_surface = not args.no_surface
    show_points = not args.no_points

    if not show_surface and not show_points and not args.impedance:
        print("Error: Nothing to visualize (surface and points disabled, no --impedance)")
        raise SystemExit(1)

    # Discover variants
    variants = discover_variants(
        model_dir, args.variant,
        need_be_out=show_surface or args.impedance or show_points,
    )

    if not variants:
        # Fallback: legacy single-variant layout
        legacy_be_out = model_dir / "numcalc" / "be.out"
        if (show_surface or show_points) and legacy_be_out.is_dir():
            variants = [None]
        else:
            if args.variant:
                print(f"Error: No variants matching: {', '.join(args.variant)}")
            else:
                print(f"Error: No variants with result data found")
            raise SystemExit(1)

    if show_points and args.set_name is None:
        offsets_files = sorted((model_dir / "numcalc").glob("*/eval_offsets.json"))
        if offsets_files:
            import json as _json
            sets = list(_json.load(open(offsets_files[0]))["sets"].keys())
            print(f"  Showing all fieldpoint sets: {', '.join(sets)} "
                  f"(pass a set name to filter)")

    # Create scene and viewer
    scene = setup_scene(screenshot=args.screenshot)
    scene._plotter.set_background(args.background)
    viewer = ResultsViewer(scene, model_dir, config, freqs, args)

    # Build initial variant
    viewer.build_variant(variants[0], bin_idx)

    if (viewer.surface_actor is None and viewer.points_actor is None
            and viewer.impedance_actor is None):
        print("Error: Nothing to visualize (surface and points both unavailable/disabled)")
        raise SystemExit(1)

    # Print actor info
    if viewer.surface_actor is not None:
        if viewer.surface_rgba:
            print(f"  + Surface pressure (phase+alpha @ {actual_freq:.1f} Hz, "
                  f"alpha=top {PHASE_DB_RANGE:.0f} dB, "
                  f"opacity={args.mesh_opacity})")
        else:
            print(f"  + Surface pressure ({viewer.surface_quantity} @ {actual_freq:.1f} Hz, "
                  f"opacity={args.mesh_opacity})")
    if viewer.points_actor is not None:
        set_label = args.set_name if args.set_name is not None else "all"
        if viewer.points_rgba:
            print(f"  + Eval points: set={set_label}, "
                  f"phase+alpha (alpha=top {PHASE_DB_RANGE:.0f} dB), "
                  f"norm={args.norm}, point_size={args.point_size}")
        else:
            print(f"  + Eval points: set={set_label}, "
                  f"quantity={viewer.points_quantity}, norm={args.norm}, "
                  f"point_size={args.point_size}")

    # Set up interactivity and show
    viewer.setup_interactivity(variants)
    scene.show_or_screenshot(args.screenshot)
