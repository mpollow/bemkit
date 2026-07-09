"""pynumcalc visualization toolbox.

Re-exports the public API from submodules.
For 3D scenes, import ``pynumcalc.viz.scene.BEMScene`` directly (lazy pyvista).
"""

from pynumcalc.viz.freq import freq_magnitude, freq_phase, freq_real_imag
from pynumcalc.viz.time import impulse_response, energy_time_curve
from pynumcalc.viz.polar import polar_directivity, polar_balloon_slice
from pynumcalc.viz.field import heatmap, contour, scatter_field, spherical_heatmap
from pynumcalc.viz._style import cmap_for_quantity
