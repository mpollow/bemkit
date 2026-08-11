"""bemkit visualization toolbox.

Re-exports the public API from submodules.
For 3D scenes, import ``bemviz.viz.scene.BEMScene`` directly (lazy pyvista).
"""

from bemviz.viz.freq import (
    freq_magnitude, freq_phase, freq_real_imag, angle_response, onaxis_response,
)
from bemviz.viz.time import (
    impulse_response, energy_time_curve, angle_impulse_response,
)
from bemviz.viz.polar import polar_directivity, polar_balloon_slice
from bemviz.viz.directivity import isobaric, directivity_levels, level_label
from bemviz.viz.field import heatmap, contour, scatter_field, spherical_heatmap
from bemviz.viz._style import cmap_for_quantity, format_polar_axes
