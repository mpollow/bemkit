"""bemviz — plotting and web export for bemkit results.

Everything that needs matplotlib or pyvista, kept out of `bemkit` so that
producing and reading results never drags a display stack in. Depends on
bemkit (for `SimulationData`/`BEMResult`), never the other way round.

The 2D plot functions all take an Axes and a `BEMResult` and return artists;
`bemviz.viz.scene.BEMScene` does 3D (lazy pyvista). `export_web` writes an
auralizer bundle.

Registers the `plot`, `geometry`, `results` and `export-web` subcommands into
the `bem` CLI through the `bemkit.cli` entry-point group — the same seam any
third-party plugin uses.
"""

from bemviz.viz import (  # noqa: F401
    angle_impulse_response, angle_response, cmap_for_quantity, contour,
    directivity_levels, energy_time_curve, format_polar_axes, freq_magnitude,
    freq_phase, freq_real_imag, heatmap, impulse_response, isobaric,
    level_label, onaxis_response, polar_balloon_slice, polar_directivity,
    scatter_field, spherical_heatmap,
)


def __getattr__(name):
    # export_web pulls in the analysis stack; keep `import bemviz` cheap.
    if name == "export_web":
        from bemviz.web import export_web
        return export_web
    raise AttributeError(f"module 'bemviz' has no attribute {name!r}")


def register_cli(sub):
    """Add bemviz's subcommands to the `bem` CLI (entry point: bemkit.cli)."""
    from bemviz import cli_plot, cli_viz3d, cli_web
    cli_plot.register(sub)
    cli_viz3d.register(sub)
    cli_web.register(sub)
