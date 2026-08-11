"""Pin the curated public API surface of the bemkit package root.

Every name exported from `bemkit` is a compatibility promise; plumbing
stays importable from its submodule but off the package root.
"""


def test_public_api_surface():
    import bemkit
    public = {n for n in dir(bemkit) if not n.startswith("_")}

    # The user API.
    assert {"ModelConfig", "freq_bin", "resolve_bins", "SimulationData",
            "export_variant", "ExportInfo", "discover_variants",
            "make_fields_onaxis", "make_fields_polar", "make_fields_plane",
            "Job", "Runner", "LocalRunner", "plan_jobs", "run_jobs",
            "available_bins", "missing_bins", "is_complete",
            "MeshInfo", "find_mesh_file", "PipelineStage",
            "normalize_pressure", "sphcap_pressure"} <= public

    # Demoted / plumbing names must be gone from the root.
    # Plotting and web export live in the separate `bemviz` distribution now;
    # bemkit must not re-export them, or it would import upward.
    for gone in ("export_web", "freq_magnitude", "polar_directivity",
                 "impulse_response", "heatmap", "cmap_for_quantity",
                 "BEMResult", "load_result", "combine_and_track",
                 "load_fieldpoint_sets", "load_offsets", "save_offsets",
                 "split_by_offsets", "validate_points", "load_bins",
                 "write_nc_inp", "write_eval_grid_nodes",
                 "write_eval_grid_elements", "write_object_mesh_nodes",
                 "write_object_mesh_elements", "read_eval_grid_nodes",
                 "read_nc_inp_boundary_conditions", "read_object_mesh_nodes",
                 "read_object_mesh_elements", "read_mesh_triangles",
                 "read_mesh_physical_groups", "element_areas",
                 "get_mesh_scale", "RIGID", "VIBRATING", "MESH_EXTENSIONS"):
        assert gone not in public, f"{gone} should not be exported at top level"


def test_core_import_stays_light():
    """`import bemkit` must not pull in matplotlib/pyvista.

    Now structural rather than a matter of discipline: bemkit does not depend
    on bemviz at all.
    """
    import subprocess
    import sys
    code = ("import sys, bemkit; "
            "bad = {m for m in ('matplotlib', 'pyvista') if m in sys.modules}; "
            "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, r.stderr.decode() or r.stdout.decode()
