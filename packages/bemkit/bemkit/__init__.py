"""bemkit — Python toolkit for Mesh2HRTF NumCalc BEM simulations.

A study is Python. You describe the problem as a :class:`~bemkit.model.Model`
(mesh, frequencies, medium, boundary conditions, sources, field points), hand
it to a solver, and read a :class:`~bemkit.results.Results` back::

    load_study(dir) -> {variant: Model} -> NumCalcSolver().solve(...) -> Results

3D inspection and web export live in :mod:`bemkit.view` (a `bemkit[view]`
extra, imported only by the CLI verbs that need it), and NumCalc's own file
formats and job control are `numcalckit`. Geometry authoring
lives in the external `bemgeo` package; bemkit consumes the STEP files it
produces.

The package root exports the user API only; plumbing (mesh generation, deck
translation, field-point validation, stores) stays importable from its
submodule. Everything below is numpy-only, so `import bemkit` never drags in
meshio, gmsh, scipy, matplotlib or pyvista — those are pulled in lazily by the
one function that needs them (``load_mesh`` needs meshio, ``sphcap_pressure``
needs scipy).
"""

import importlib

# -- The problem (pure numpy data, no I/O, no solver) --
from bemkit.model import (
    BC,
    Admittance,
    Condition,
    Elements,
    Group,
    Impedance,
    Medium,
    Mesh,
    Model,
    Near,
    PlaneWave,
    PointSource,
    Pressure,
    ResolvedBC,
    Selector,
    Velocity,
    Where,
    circle,
    fft_grid,
    on_axis,
    plane,
    resolve_bins,
)

# -- Getting a Mesh off disk --
from bemkit.geometry.io import MeshInfo, find_mesh_file
from bemkit.geometry.load import load_mesh

# -- Solving, and what comes back --
from bemkit.study import load_study
from bemkit.results import MemoryStore, NpzStore, Results, Store
from bemkit.solvers import Solver, Status
from bemkit.solvers.bempp import BemppSolver
from bemkit.solvers.numcalc import NumCalcSolver

# -- Job control, straight through from numcalckit --
from numcalckit.runner import Job, LocalRunner, Runner, plan_jobs, run_jobs
from numcalckit.status import available_bins, is_complete, missing_bins

# -- Lazy: needs scipy, which is an optional extra (`pip install bemkit[analytic]`) --
_LAZY = {
    "sphcap_pressure": "bemkit.model.analytic",
}

__all__ = [
    # model
    "Mesh", "Medium", "Model", "BC", "ResolvedBC",
    "Selector", "Group", "Elements", "Near", "Where",
    "Condition", "Velocity", "Pressure", "Admittance", "Impedance",
    "PlaneWave", "PointSource",
    "fft_grid", "on_axis", "circle", "plane", "resolve_bins",
    # geometry
    "load_mesh", "MeshInfo", "find_mesh_file",
    # study / solve / results
    "load_study", "Solver", "Status", "NumCalcSolver", "BemppSolver",
    "Results", "Store", "MemoryStore", "NpzStore",
    # job control
    "Job", "Runner", "LocalRunner", "plan_jobs", "run_jobs",
    "available_bins", "missing_bins", "is_complete",
    # lazy
    "sphcap_pressure",
]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'bemkit' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
