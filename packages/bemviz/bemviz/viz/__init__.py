"""3D rendering helpers.

Import ``bemviz.viz.scene.BEMScene`` directly for scenes (lazy pyvista).

``results_viewer.ResultsViewer`` — the interactive viewer behind `bem MODEL
results` — was deleted with the config.toml stack it was built on; see
``bemviz/__init__.py`` for where to find it in git history.
"""

from bemviz.viz._style import cmap_for_quantity  # noqa: F401
