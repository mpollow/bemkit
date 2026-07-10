"""bemkit — Python toolkit for Mesh2HRTF NumCalc BEM simulations.

Layers: meshing / sim (core) and analysis / viz (results).
Geometry authoring lives in the external `bemgeo` package; bemkit consumes
the STEP files it produces.
Core imports eagerly and never needs matplotlib or pyvista; the analysis
and viz names below resolve lazily on first attribute access.
"""

import importlib

# -- Core (eager) --
from bemkit.config import ModelConfig, freq_bin
from bemkit.meshing.io import (
    RIGID, VIBRATING, MESH_EXTENSIONS, MeshInfo,
    find_mesh_file, get_mesh_scale, read_mesh_triangles,
    read_mesh_physical_groups, element_areas,
)
from bemkit.pipeline import PipelineStage
from bemkit.sim.discovery import discover_variants
from bemkit.sim.export import ExportInfo, export_variant
from bemkit.sim.fields import make_fields_onaxis, make_fields_plane, make_fields_polar
from bemkit.sim.fieldpoints import (
    combine_and_track, load_fieldpoint_sets, load_offsets,
    save_offsets, split_by_offsets, validate_points,
)
from bemkit.sim.readers import (
    load_bins, read_eval_grid_nodes, read_nc_inp_boundary_conditions,
    read_object_mesh_elements, read_object_mesh_nodes,
)
from bemkit.sim.runner import Job, LocalRunner, Runner, plan_jobs, run_jobs
from bemkit.sim.status import available_bins, is_complete, missing_bins
from bemkit.sim.writers import (
    write_eval_grid_elements, write_eval_grid_nodes, write_nc_inp,
    write_object_mesh_elements, write_object_mesh_nodes,
)

# -- Analysis / viz (lazy: `import bemkit` stays matplotlib-free) --
_LAZY = {
    "export_web": "bemkit.web",
    "SimulationData": "bemkit.analysis.simulation",
    "BEMResult": "bemkit.analysis.result",
    "load_result": "bemkit.analysis.result",
    "normalize_pressure": "bemkit.analysis.normalization",
    "sphcap_pressure": "bemkit.analysis.analytic",
    "freq_magnitude": "bemkit.viz",
    "freq_phase": "bemkit.viz",
    "freq_real_imag": "bemkit.viz",
    "impulse_response": "bemkit.viz",
    "energy_time_curve": "bemkit.viz",
    "polar_directivity": "bemkit.viz",
    "polar_balloon_slice": "bemkit.viz",
    "heatmap": "bemkit.viz",
    "contour": "bemkit.viz",
    "scatter_field": "bemkit.viz",
    "spherical_heatmap": "bemkit.viz",
    "cmap_for_quantity": "bemkit.viz",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'bemkit' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
