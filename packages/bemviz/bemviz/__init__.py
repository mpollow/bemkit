"""bemviz — 3D inspection and web export for bemkit results.

Everything that needs pyvista or matplotlib, kept out of `bemkit` so that
producing and reading results never drags a display stack in. Depends on
bemkit, never the other way round.

`bemviz.viz.scene.BEMScene` renders 3D scenes (lazy pyvista); `export_web`
writes an auralizer bundle.

Registers the `geometry`, `results` and `export-web` subcommands into the `bem`
CLI through the `bemkit.cli` entry-point group — the same seam any third-party
plugin uses.

**Half-ported.** `geometry` runs on the model.py spine. `results` and
`export-web` do not yet: both were built on `SimulationData`/`BEMResult`,
which the spine deliberately dropped, so both refuse with a message naming
what is missing rather than failing somewhere inside. Folding this
distribution into `bemkit/view/` is the task that finishes them; the
interactive `ResultsViewer` was deleted with the config.toml stack on
2026-08-12 and is the port's starting point in git history
(`git show HEAD~1:packages/bemviz/bemviz/viz/results_viewer.py`).

The 2D plotting family (`bem MODEL plot`: freq/ir/polar/isobaric/...) was
removed on 2026-08-12; recover it from git history if it is wanted back.
"""

from bemviz.viz._style import cmap_for_quantity  # noqa: F401


def __getattr__(name):
    # export_web pulls in the analysis stack; keep `import bemviz` cheap.
    if name == "export_web":
        from bemviz.web import export_web
        return export_web
    raise AttributeError(f"module 'bemviz' has no attribute {name!r}")


def register_cli(sub):
    """Add bemviz's subcommands to the `bem` CLI (entry point: bemkit.cli)."""
    from bemviz import cli_viz3d, cli_web
    cli_viz3d.register(sub)
    cli_web.register(sub)
