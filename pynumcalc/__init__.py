"""pynumcalc — Python toolkit for Mesh2HRTF NumCalc BEM simulations.

Layers: meshing / sim (core) and analysis / viz (results).
Geometry authoring lives in the external `bemgeo` package; pynumcalc consumes
the STEP files it produces.
Core imports eagerly and never needs matplotlib or pyvista; the analysis
and viz names below resolve lazily on first attribute access.
"""

import importlib

# -- Core (eager) --
from pynumcalc.config import ModelConfig, freq_bin
from pynumcalc.meshing.io import (
    RIGID, VIBRATING, MESH_EXTENSIONS, MeshInfo,
    find_mesh_file, get_mesh_scale, read_mesh_triangles,
    read_mesh_physical_groups, element_areas,
)
from pynumcalc.pipeline import PipelineStage
from pynumcalc.sim.discovery import discover_variants
from pynumcalc.sim.export import ExportInfo, export_variant
from pynumcalc.sim.fields import make_fields_onaxis, make_fields_plane, make_fields_polar
from pynumcalc.sim.fieldpoints import (
    combine_and_track, load_fieldpoint_sets, load_offsets,
    save_offsets, split_by_offsets, validate_points,
)
from pynumcalc.sim.readers import (
    load_bins, read_eval_grid_nodes, read_nc_inp_boundary_conditions,
    read_object_mesh_elements, read_object_mesh_nodes,
)
from pynumcalc.sim.runner import Job, LocalRunner, Runner, plan_jobs, run_jobs
from pynumcalc.sim.status import available_bins, is_complete, missing_bins
from pynumcalc.sim.writers import (
    write_eval_grid_elements, write_eval_grid_nodes, write_nc_inp,
    write_object_mesh_elements, write_object_mesh_nodes,
)

# -- Analysis / viz (lazy: `import pynumcalc` stays matplotlib-free) --
_LAZY = {
    "SimulationData": "pynumcalc.analysis.simulation",
    "BEMResult": "pynumcalc.analysis.result",
    "load_result": "pynumcalc.analysis.result",
    "normalize_pressure": "pynumcalc.analysis.normalization",
    "sphcap_pressure": "pynumcalc.analysis.analytic",
    "freq_magnitude": "pynumcalc.viz",
    "freq_phase": "pynumcalc.viz",
    "freq_real_imag": "pynumcalc.viz",
    "impulse_response": "pynumcalc.viz",
    "energy_time_curve": "pynumcalc.viz",
    "polar_directivity": "pynumcalc.viz",
    "polar_balloon_slice": "pynumcalc.viz",
    "heatmap": "pynumcalc.viz",
    "contour": "pynumcalc.viz",
    "scatter_field": "pynumcalc.viz",
    "spherical_heatmap": "pynumcalc.viz",
    "cmap_for_quantity": "pynumcalc.viz",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'pynumcalc' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
