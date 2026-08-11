"""bemkit — Python toolkit for Mesh2HRTF NumCalc BEM simulations.

Modelling and results on top of `pynumcalc`: meshing, fieldpoints, export,
run orchestration, and the `SimulationData` results API.

Plotting is the separate `bemviz` distribution, and NumCalc's own file formats
and job control are `pynumcalc`. Geometry authoring lives in the external
`bemgeo` package; bemkit consumes the STEP files it produces.

The package root exports the user API only; plumbing (file readers/writers,
fieldpoint bookkeeping, mesh I/O internals) stays importable from its
submodule. Core imports eagerly; the analysis names below resolve lazily.
"""

import importlib

# -- Core (eager) --
from bemkit.config import ModelConfig, freq_bin, resolve_bins
from bemkit.meshing.io import MeshInfo, find_mesh_file
from bemkit.pipeline import PipelineStage
from bemkit.sim.discovery import discover_variants
from bemkit.sim.export import ExportInfo, export_variant
from bemkit.sim.fields import make_fields_onaxis, make_fields_plane, make_fields_polar
from pynumcalc.runner import Job, LocalRunner, Runner, plan_jobs, run_jobs
from pynumcalc.status import available_bins, is_complete, missing_bins

# -- Analysis (lazy: `import bemkit` stays import-light) --
# Plotting lives in the separate `bemviz` distribution; bemkit never imports it.
_LAZY = {
    "SimulationData": "bemkit.analysis.simulation",
    "normalize_pressure": "bemkit.analysis.normalization",
    "sphcap_pressure": "bemkit.analysis.analytic",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'bemkit' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
