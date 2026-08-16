"""NumCalc format and process layer — everything the solver itself understands.

This subpackage is the boundary between *what NumCalc reads* and *what bemkit
decides*. It knows the ``NC.inp`` deck grammar, the ``be.out/`` result format, and
how to invoke and track the solver binary. It knows nothing about studies,
Models, mesh files, variants, or any other bemkit concept.

**It must not import anything from ``bemkit``** — only stdlib and numpy. The rule
is enforced by ``tests/test_boundaries.py``; ``bemkit.solvers`` may import from here,
never the reverse. Keeping that direction true is what leaves the door open to
publishing this layer as a standalone package (see
``docs/numcalc-feature-coverage.md`` §4).

Typical use — write a deck, run it, read the results back::

    from numcalckit import Deck, BC, Job, LocalRunner, load_bins

    deck = Deck(title="sphere", frequencies=np.array([250.0, 500.0]),
                n_total_nodes=..., n_total_elements=...,
                boundary=[BC(0, 111, "VELO", 1.0)])
    deck.write(nc_dir / "NC.inp")
    run_jobs([Job("sphere", nc_dir)])          # runner defaults to LocalRunner
    p = load_bins(nc_dir, "pBoundary", len(deck.frequencies))

The names below are this layer's public surface; everything else is internal.
"""

from numcalckit.deck import (
    BC, BC_KINDS, Curve, Deck, Source, parse_bc_line,
)
from numcalckit.meshfiles import (
    write_eval_grid_elements, write_eval_grid_nodes, write_object_mesh,
)
from numcalckit.readers import (
    QUANTITIES, list_bins, load_bins, read_eval_grid_nodes,
    read_nc_inp_boundary_conditions, read_nc_inp_symmetry,
    read_object_mesh_elements, read_object_mesh_nodes,
)
from numcalckit.packing import (
    clear_pack, has_results, is_packed, pack_variant, unpack_variant,
)
from numcalckit.status import (
    CGS_TOLERANCE, NITER_MAX_DEFAULT, BinRecord, available_bins, convergence,
    is_complete, missing_bins, numcalc_pids, scan_numcalc, solver_warnings,
)
from numcalckit.runner import (
    Job, LocalRunner, Runner, plan_jobs, run_jobs,
)

__all__ = [
    # deck
    "Deck", "BC", "Source", "Curve", "BC_KINDS", "parse_bc_line",
    # mesh files
    "write_object_mesh", "write_eval_grid_nodes", "write_eval_grid_elements",
    # readers
    "load_bins", "list_bins", "QUANTITIES", "read_eval_grid_nodes",
    "read_object_mesh_nodes", "read_object_mesh_elements",
    "read_nc_inp_boundary_conditions", "read_nc_inp_symmetry",
    # packing
    "pack_variant", "unpack_variant", "clear_pack", "is_packed", "has_results",
    # status
    "available_bins", "missing_bins", "is_complete", "solver_warnings",
    "numcalc_pids", "scan_numcalc",
    "convergence", "BinRecord", "CGS_TOLERANCE", "NITER_MAX_DEFAULT",
    # running
    "Job", "Runner", "LocalRunner", "plan_jobs", "run_jobs",
]
