"""bemkit.view — 3D inspection and web export.

Everything that needs pyvista or matplotlib, kept in one layer so that
producing and reading results never drags a display stack in: nothing under
``model/``, ``geometry/``, ``solvers/`` or ``results/`` may import from here,
and ``tests/test_boundaries.py`` enforces it.

``scene.BEMScene`` renders 3D scenes (lazy pyvista), ``results_viewer``
drives the interactive `bem STUDY results` viewer on top of it, and
``export_web`` writes an auralizer bundle.

Adds the `geometry`, `results` and `export-web` subcommands to the `bem` CLI.
Registration is a direct, lazily-imported call from ``bemkit.cli`` (see
``main``), not an entry point: this used to be the separate ``bemviz``
distribution, and the ``bemkit.cli`` entry-point group it went through is now
only what the private ``bemlab`` package uses.

``export-web`` still refuses: it was written against the config.toml stack and
has not been ported to the model.py spine, so it says which piece is missing
rather than failing somewhere inside.

The 2D plotting family (`bem MODEL plot`: freq/ir/polar/isobaric/...) was
removed on 2026-08-12; recover it from git history if it is wanted back.
"""

from bemkit.view._style import cmap_for_quantity  # noqa: F401


def __getattr__(name):
    # export_web pulls in the whole bundle writer; keep the import cheap.
    if name == "export_web":
        from bemkit.view.web import export_web
        return export_web
    raise AttributeError(f"module 'bemkit.view' has no attribute {name!r}")


def register_cli(sub):
    """Add the view subcommands to the `bem` CLI."""
    from bemkit.view import cli_viz3d, cli_web
    cli_viz3d.register(sub)
    cli_web.register(sub)
