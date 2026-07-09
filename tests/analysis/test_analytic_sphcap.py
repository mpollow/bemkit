"""Validation tests: BEM (NumCalc) vs analytic solution for spherical cap.

Compares pEvalGrid and pBoundary output from the sphcap-50-20.lc5-5 variant
against the exact spherical-harmonic analytic solution.  Tests are skipped
automatically when the BEM data is not present.
"""

import numpy as np
import pytest
from pathlib import Path

from pynumcalc.analysis.analytic import sphcap_pressure
from pynumcalc.sim.readers import (
    read_eval_grid_nodes,
    read_object_mesh_nodes,
    read_object_mesh_elements,
    load_bins,
)

SPHCAP_DIR  = Path(__file__).parent.parent / "models" / "sphcap"
VARIANT     = "sphcap-50-20.lc5-5"
NUMCALC_DIR = SPHCAP_DIR / "numcalc" / VARIANT

RADIUS_MM = 50.0
ANGLE_DEG = 20.0
RHO0      = 1.2
C0        = 340.0
SR        = 32000
NBINS     = 65

pytestmark = pytest.mark.skipif(
    not NUMCALC_DIR.exists(),
    reason=f"{VARIANT} BEM data not found at {NUMCALC_DIR}",
)


@pytest.fixture(scope="module")
def data():
    freqs = np.linspace(0, SR / 2, NBINS)[1:]
    nmax  = max(8, int(2 * np.pi * (RADIUS_MM / 1000) / (C0 / freqs.max())) + 2)

    nf = NBINS - 1

    eval_pts  = read_eval_grid_nodes(NUMCALC_DIR)
    bem_field = load_bins(NUMCALC_DIR, "pEvalGrid", nf)

    nodes     = read_object_mesh_nodes(NUMCALC_DIR)
    elements  = read_object_mesh_elements(NUMCALC_DIR)
    centroids = nodes[elements].mean(axis=1)
    surf_pts  = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * (RADIUS_MM / 1000)
    bem_surf  = load_bins(NUMCALC_DIR, "pBoundary", nf)

    kw = dict(c0=C0, rho0=RHO0, angle_deg=ANGLE_DEG, radius_mm=RADIUS_MM, nmax=nmax)
    return freqs, bem_field, bem_surf, \
           sphcap_pressure(eval_pts, freqs, **kw), \
           sphcap_pressure(surf_pts, freqs, **kw)


def _mean_rel_err(a, b):
    return (np.abs(a - b) / (np.abs(b) + 1e-30)).mean()


def test_field_pressure(data):
    freqs, bem_field, _, analytic_field, _ = data
    low = freqs <= 2000
    assert _mean_rel_err(analytic_field[:, low], bem_field[:, low]) < 0.05


def test_surface_pressure(data):
    freqs, _, bem_surf, _, analytic_surf = data
    low = freqs <= 2000
    assert _mean_rel_err(analytic_surf[:, low], bem_surf[:, low]) < 0.05
