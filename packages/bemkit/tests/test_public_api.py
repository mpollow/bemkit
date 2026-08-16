"""Pin the curated public API surface of the bemkit package root.

Every name exported from `bemkit` is a compatibility promise; plumbing
(deck translation, field-point validation, mesh generation, stores) stays
importable from its submodule but off the package root.
"""

import subprocess
import sys

import pytest


def test_public_api_surface():
    import bemkit
    public = {n for n in dir(bemkit) if not n.startswith("_")}

    # The user API: describe a problem, load it, solve it, read it back.
    assert {
        # the problem
        "Mesh", "Medium", "Model", "BC", "ResolvedBC",
        "Selector", "Group", "Elements", "Near", "Where",
        "Condition", "Velocity", "Pressure", "Admittance", "Impedance",
        "PlaneWave", "PointSource",
        "fft_grid", "on_axis", "circle", "plane", "resolve_bins",
        # getting a mesh off disk
        "load_mesh", "MeshInfo", "find_mesh_file",
        # study / solve / results
        "load_study", "Solver", "Status", "NumCalcSolver", "BemppSolver",
        "Results", "Store", "MemoryStore", "NpzStore",
        # job control, straight through from numcalckit
        "Job", "Runner", "LocalRunner", "plan_jobs", "run_jobs",
        "available_bins", "missing_bins", "is_complete",
        # analytic reference (lazy — needs scipy)
        "sphcap_pressure",
    } <= public


def test_every_advertised_name_actually_resolves():
    """__all__ is the promise; dir() alone would not catch a lazy name whose
    target module moved."""
    import bemkit
    for name in bemkit.__all__:
        assert getattr(bemkit, name) is not None, name


def test_unknown_attribute_still_raises():
    import bemkit
    with pytest.raises(AttributeError):
        bemkit.no_such_name


def test_the_deleted_config_toml_stack_is_gone():
    """The TOML grammar, the stage machine and the SimulationData results API
    were deleted with the config.toml stack. They must not come back as
    aliases on the root — a name that resolves is a name people will use."""
    import bemkit
    public = {n for n in dir(bemkit) if not n.startswith("_")}
    for gone in ("ModelConfig", "freq_bin", "freq_grid", "PipelineStage",
                 "SimulationData", "BEMResult", "load_result",
                 "normalize_pressure", "discover_variants", "export_variant",
                 "ExportInfo", "make_fields_onaxis", "make_fields_polar",
                 "make_fields_plane"):
        assert gone not in public, f"{gone} should no longer exist"

    for mod in ("bemkit.config", "bemkit.sim", "bemkit.pipeline",
                "bemkit.analysis"):
        with pytest.raises(ImportError):
            __import__(mod)


def test_demoted_plumbing_stays_off_the_root():
    """Importable from its submodule, not from the package root."""
    import bemkit
    public = {n for n in dir(bemkit) if not n.startswith("_")}
    for gone in ("export_web", "cmap_for_quantity", "BEMScene",
                 "model_to_deck", "Deck", "load_bins", "write_object_mesh",
                 "read_mesh_triangles", "read_mesh_physical_groups",
                 "element_areas", "get_mesh_scale",
                 "RIGID", "VIBRATING", "MESH_EXTENSIONS"):
        assert gone not in public, f"{gone} should not be exported at top level"


@pytest.mark.parametrize("heavy", ["matplotlib", "pyvista", "meshio", "gmsh",
                                   "scipy", "bempp_cl"])
def test_core_import_stays_light(heavy):
    """`import bemkit` must pull in nothing but numpy and numcalckit.

    Every heavy dependency — meshio, gmsh, scipy, and since the bemviz fold
    the display stack too — lives behind a lazy import inside the one function
    that needs it. The display stack used to be a separate distribution, so
    packaging enforced this on its own; now that ``bemkit.view`` ships inside
    bemkit, this test is the only thing that does.
    """
    code = (f"import sys, bemkit; "
            f"sys.exit({heavy!r} if {heavy!r} in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"import bemkit pulled in {heavy}: "
        f"{r.stderr.decode() or r.stdout.decode()}")


@pytest.mark.parametrize("heavy", ["matplotlib", "pyvista"])
def test_importing_the_view_layer_is_itself_cheap(heavy):
    """Even `import bemkit.view` must not start a display stack.

    `bem --help` builds every subparser, which imports this package to
    register the view verbs. If pyvista were imported at module level there,
    every `bem` invocation would pay several seconds for it.
    """
    code = (f"import sys, bemkit.view; "
            f"sys.exit({heavy!r} if {heavy!r} in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"import bemkit.view pulled in {heavy}: "
        f"{r.stderr.decode() or r.stdout.decode()}")


@pytest.mark.parametrize("heavy", ["scipy", "meshio", "gmsh"])
def test_the_model_layer_alone_is_lighter_still(heavy):
    """`import bemkit.model` is the bottom of the spine: numpy only, so a
    Model can be built and compared with nothing else installed. analytic.py
    needs scipy, which is why it hangs off a lazy __getattr__."""
    code = (f"import sys, bemkit.model; "
            f"sys.exit({heavy!r} if {heavy!r} in sys.modules else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"import bemkit.model pulled in {heavy}: "
        f"{r.stderr.decode() or r.stdout.decode()}")


def test_sphcap_pressure_is_reachable_from_the_root_and_the_model_layer():
    import bemkit
    import bemkit.model
    from bemkit.model.analytic import sphcap_pressure
    assert bemkit.sphcap_pressure is sphcap_pressure
    assert bemkit.model.sphcap_pressure is sphcap_pressure
