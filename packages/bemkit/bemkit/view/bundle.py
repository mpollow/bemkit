"""Finding, preparing and reading the run bundles a viewer draws.

A viewer needs three things a :class:`~bemkit.results.Results` deliberately
does not carry: the boundary mesh, the evaluation-point coordinates, and the
boundary conditions that produced the numbers. ``Results`` is store-backed and
solver-neutral — a ``MemoryStore``-backed one from ``solve(workdir=None)`` has
no files at all — so the geometry is read back out of the run bundle
(``STUDY/runs/<solver>/<variant>/``) through :mod:`numcalckit`'s readers, which
is exactly what NumCalc was given.

Reading NC.inp rather than the Model for the boundary conditions is
deliberate in the same way: what is drawn should be what the solver actually
saw, so a stale bundle looks stale instead of silently showing the model.py
you have now.

No pyvista here — this module is the testable half of the view layer.
"""

import sys
from pathlib import Path

import numpy as np

from bemkit.results.normalization import acceleration_scale
from bemkit.results.results import Results
from bemkit.study import load_study, match_variants, workdir


def _solver():
    """The one solver backend there is. Its name is the runs/ level."""
    from bemkit.solvers.numcalc import NumCalcSolver
    return NumCalcSolver()


def prepared_bundles(study_dir, prefixes, *, require_results=False):
    """``(names, workdir_for)`` for the selected variants of a study.

    Exports any variant whose bundle is missing. There is no separate `bem
    export` command on the model.py spine — `estimate-ram` already exports
    what it needs for the same reason — so demanding a prepared bundle here
    would mean "run the solve you are trying to inspect first". Exporting is
    cheap, deterministic, and writes nothing a later solve depends on; a
    variant with a live NumCalc is left exactly as it is.

    Args:
        study_dir: the study directory.
        prefixes: ``--variant`` prefix filter, or None for all.
        require_results: also skip variants with no solved bins. `results`
            passes True — a run directory with only an export in it has
            nothing to colour, and saying so per variant beats an empty
            window.

    Exits (SystemExit) when nothing is left to show.
    """
    from numcalckit.status import numcalc_pids

    models = load(study_dir)
    try:
        names = match_variants(models, prefixes)
    except KeyError as e:
        sys.exit(e.args[0])
    solver = _solver()

    ready = []
    for name in names:
        wd = workdir(study_dir, name, solver.name)
        # Checked before exporting, not after: a variant with no results is
        # not going to be drawn, and exporting it first would mesh and write
        # a whole NC.inp bundle for nothing.
        if require_results and not has_results(wd):
            print(f"  {name}: skipping (no solved bins yet)")
            continue
        exported = ((wd / "ObjectMesh" / "Nodes.txt").exists()
                    and (wd / "NC.inp").exists())
        if not exported:
            if numcalc_pids(wd):
                print(f"  {name}: skipping (NumCalc still running)")
                continue
            try:
                solver.export(models[name], wd)
            except ValueError as e:
                print(f"  {name}: cannot export — {e}")
                continue
        ready.append(name)

    if not ready:
        what = "with results" if require_results else "inspectable"
        sys.exit(f"No {what} variant in {study_dir} "
                 f"(tried: {', '.join(names)})\n"
                 f"  Solve one first:  bem {study_dir} solve")
    return ready, models, lambda v: workdir(study_dir, v, solver.name)


def load(study_dir):
    """``load_study`` with a viewer-friendly error for the missing-study case."""
    try:
        return load_study(study_dir)
    except FileNotFoundError as e:
        sys.exit(f"{e}\n"
                 f"  A study is a directory holding a model.py that defines "
                 f"build() -> Model (or variants() -> {{name: Model}}).")


def has_results(wd):
    """True when ``wd`` holds anything a viewer could colour by."""
    from numcalckit.packing import has_results as _raw
    return (wd / "results.npz").exists() or _raw(wd)


# ── reading one bundle ─────────────────────────────────────────────────────


def load_results(wd, model):
    """The Results for a bundle, preferring ``results.npz`` when it is current.

    ``results.npz`` is what a finished solve leaves behind and is far cheaper
    to read than a ``be.out/`` of ASCII files. It can also be *stale*: a
    hand-run ``NumCalc``, or a resume still in flight, adds bins underneath it
    without rewriting it. So the raw output wins whenever it is newer, and
    ``Results.from_beout`` refreshes the npz as a side effect.

    The mtime comparison is against ``be.out/`` itself (or ``be.npz``): a new
    bin means a new ``be.out/be.N/`` subdirectory, which bumps the parent's
    mtime. It cannot see a bin being *rewritten* in place, which only --force
    does, and --force deletes be.out/ first.
    """
    npz = wd / "results.npz"
    raw = wd / "be.npz"
    if not raw.exists():
        raw = wd / "be.out"
    if npz.exists() and (not raw.exists()
                         or npz.stat().st_mtime >= raw.stat().st_mtime):
        return Results.load(npz)
    return Results.from_beout(wd, model.frequencies, _offsets(wd, model))


def _offsets(wd, model):
    """Field-point offsets for a bundle: the file the export wrote, else the
    model's own sets in emission order.

    The file is authoritative because validation may have dropped points, so
    a set's rows are not simply its length; falling back to the model is for
    a hand-assembled bundle and is better than refusing to read one."""
    import json

    path = wd / "eval_offsets.json"
    if path.exists():
        with open(path) as f:
            return {k: np.asarray(v, dtype=int) for k, v in json.load(f).items()}
    offsets, start = {}, 0
    for name, pts in model.fields.items():
        offsets[name] = np.arange(start, start + len(pts))
        start += len(pts)
    return offsets


def computed_bins(results, quantities=("pEvalGrid", "pBoundary")):
    """0-based frequency indices that actually hold numbers.

    A partial run (`bem STUDY solve --preview 6`) leaves the other bins as
    NaN — ``numcalckit.load_bins`` fills them that way precisely so a missing
    bin is distinguishable from one that solved to zero. A viewer must never
    offer them: NaN reaches the colour mapper as "below the range" and paints
    a confident, wrong colour.

    Returns:
        A sorted int array. Empty when nothing has been solved.
    """
    nf = len(results.frequencies)
    seen = np.zeros(nf, dtype=bool)
    for q in quantities:
        try:
            arr = results.store.get(q)
        except KeyError:
            continue
        # Any finite value in a bin means NumCalc wrote that bin.
        axes = tuple(i for i in range(arr.ndim) if i != 1)
        seen |= np.isfinite(arr).any(axis=axes)
    return np.flatnonzero(seen)


def nearest_bin(frequencies, target_hz, allowed=None):
    """Index of the bin nearest ``target_hz``, restricted to ``allowed``.

    ``allowed`` is what makes a partial band usable: asking for 1000 Hz in a
    six-bin preview must land on a bin that was computed, not on the one that
    would have been nearest had everything been solved.

    Returns:
        A 0-based index into ``frequencies``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    idx = (np.arange(len(frequencies)) if allowed is None
           else np.asarray(allowed, dtype=int))
    if len(idx) == 0:
        raise ValueError("no bins to choose from")
    return int(idx[np.argmin(np.abs(frequencies[idx] - float(target_hz)))])


class Bundle:
    """One variant's drawable state: geometry from disk, numbers from Results.

    Loaded once per variant and reused across frequency steps — reloading it
    per frame is what made the old viewer cost ~0.5 s a step instead of
    ~0.05 s.
    """

    def __init__(self, wd, model, *, norm="raw", mirror=True):
        from numcalckit.readers import (
            read_eval_grid_nodes, read_nc_inp_boundary_conditions,
            read_nc_inp_symmetry, read_object_mesh_elements,
            read_object_mesh_nodes,
        )

        self.wd = Path(wd)
        self.model = model
        self.nodes = read_object_mesh_nodes(self.wd)
        self.elements = read_object_mesh_elements(self.wd)
        self.eval_points = read_eval_grid_nodes(self.wd)
        self.bc_elements = read_nc_inp_boundary_conditions(self.wd / "NC.inp")
        self.symmetry = read_nc_inp_symmetry(self.wd / "NC.inp")

        raw = load_results(self.wd, model)
        self.frequencies = raw.frequencies
        self.norm = norm
        # Raises NoVelocityDrive for a scattering model. Deliberately not
        # caught here: a library object must raise so a caller can catch,
        # and cmd_results turns it into a message.
        results = (raw if norm == "raw"
                   else raw.scaled_by(acceleration_scale(model)))
        # Apply the scale once rather than on every read: a frequency step
        # re-reads the whole boundary array, and multiplying a 36k x 64
        # complex array per frame is a visible cost for no gain.
        self.results = results.materialise()
        self.bins = computed_bins(self.results)

        self.ghost = None
        if mirror:
            from bemkit.view._symmetry import ghost_geometry
            self.ghost = ghost_geometry(self.nodes, self.elements, self.symmetry)

    # -- per-set access ----------------------------------------------------

    @property
    def set_names(self):
        return self.results.set_names

    def points(self, set_name=None):
        """(N, 3) coordinates for one field-point set, or all of them.

        Indexed through the same offsets ``Results`` splits its rows by, so
        point *i* of ``points(name)`` is row *i* of ``pressure(name)``.
        """
        if set_name is None:
            return self.eval_points
        return self.eval_points[self.results.offsets[set_name]]

    def pressure(self, bin_index, set_name=None):
        """(N,) complex eval-point pressure at one 0-based bin."""
        return self.results.pressure(set_name)[:, bin_index]

    def velocity(self, bin_index, set_name=None):
        """(N, 3) complex eval-point velocity at one 0-based bin."""
        return self.results.velocity(set_name)[:, bin_index]

    def boundary_pressure(self, bin_index):
        """(M,) complex surface pressure at one 0-based bin."""
        return self.results.boundary_pressure()[:, bin_index]

    def has(self, quantity):
        return quantity in self.results.store.available()
