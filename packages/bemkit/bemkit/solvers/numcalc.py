"""Model -> NumCalc Deck, and the Solver that drives NumCalc end to end.

This is the single place NumCalc-isms are allowed to live. The model layer
above (:mod:`bemkit.model`) knows nothing of decks, element ranges or CURVES;
:mod:`numcalckit` below knows nothing of models. Everything opinionated about
*how* a solver-neutral :class:`~bemkit.model.Model` becomes an ``NC.inp`` deck
belongs here — ``model_to_deck`` for the translation, :class:`NumCalcSolver`
for the write-run-ingest pipeline built on top of it.
"""

import json
import tempfile
from pathlib import Path

from numcalckit import (
    available_bins, missing_bins, numcalc_pids, plan_jobs, run_jobs,
    solver_warnings,
)
from numcalckit.deck import BC as DeckBC
from numcalckit.deck import Curve, Deck, Source, _contiguous_ranges
from numcalckit.meshfiles import (
    write_eval_grid_elements, write_eval_grid_nodes, write_object_mesh,
)

from bemkit.model.sources import PlaneWave, PointSource
from bemkit.results.results import Results
from bemkit.solvers import fieldpoints
from bemkit.solvers.base import Status

#: NumCalc supports frequency CURVES for ADMI only: it ignores them for
#: VELO/PRES and exit(-1)s on IMPE with a curve, telling you to use ADMI
#: instead. That is physics, not an arbitrary limitation — admittance enters
#: the boundary *operator*, so the system matrix genuinely changes per
#: frequency, whereas an excitation is a scale factor on the answer and is
#: handled by post-scaling the results (``Results.scaled_by``) instead.
_CURVE_KINDS = {"ADMI"}


def _boundary_lines(model):
    """Resolve every BC into Deck BC records plus the CURVES they need.

    Raises if:
      * a BC's selector resolves to no elements (an empty physical group or
        an empty literal list) — such a BC would otherwise vanish from the
        deck without a trace, which is never what a model author intended;
      * two BCs assign the *same* NumCalc kind to the same element. NumCalc
        reads BOUNDARY lines in file order, and a later line for an element
        that already has one silently overwrites the earlier value — so two
        overlapping same-kind BCs can never be valid input, one would just
        be discarded. A VELO and an ADMI sharing an element is unaffected:
        that combination is documented, valid NumCalc input.
    """
    freqs = model.frequencies
    boundary, curves = [], []
    assigned = {}  # element index -> set of kinds already used there

    for rbc in model.resolved_bcs():
        cond = rbc.condition
        kind = cond.kind
        elements = sorted({int(x) for x in rbc.elements})
        if not elements:
            raise ValueError(
                f"a {kind} boundary condition selects no elements — check "
                f"the selector (an empty physical group, or an empty "
                f"literal list, resolves to nothing)")

        for el in elements:
            kinds_here = assigned.setdefault(el, set())
            if kind in kinds_here:
                raise ValueError(
                    f"element {el} has two {kind} boundary conditions "
                    f"assigned. NumCalc reads BOUNDARY lines in file order "
                    f"and a later line for an element that already has one "
                    f"silently overwrites the earlier value, so this is "
                    f"always a modelling mistake, not a valid combination.")
            kinds_here.add(kind)

        if cond.is_frequency_dependent:
            if kind not in _CURVE_KINDS:
                raise ValueError(
                    f"NumCalc cannot vary a {kind} condition with frequency "
                    f"(it supports CURVES for ADMI only — VELO/PRES ignore "
                    f"a curve and IMPE hard-exits on one). For an "
                    f"excitation this is unnecessary anyway: solve at unit "
                    f"amplitude and use Results.scaled_by() instead of "
                    f"re-solving per frequency.")
            values = cond.at(freqs)
            cid = len(curves) + 1
            points_re = list(zip((float(f) for f in freqs),
                                 (float(v) for v in values.real)))
            points_im = list(zip((float(f) for f in freqs),
                                 (float(v) for v in values.imag)))
            curves.append(Curve(cid, points_re))
            curves.append(Curve(cid + 1, points_im))
            # NumCalc multiplies each part by its own curve, so both
            # constants on the ELEM line must be 1.0 — a 0.0 imaginary
            # constant would silently zero the imaginary curve.
            real, imag, curve_real, curve_imag = 1.0, 1.0, cid, cid + 1
        else:
            v = complex(cond.at(freqs)[0])
            real, imag, curve_real, curve_imag = v.real, v.imag, -1, -1

        for start, end in _contiguous_ranges(elements):
            boundary.append(DeckBC(start=start, end=end, kind=kind,
                                   real=real, imag=imag,
                                   curve_real=curve_real,
                                   curve_imag=curve_imag))

    return boundary, curves


def _sources(model):
    """PlaneWave/PointSource -> Deck Source records.

    ``nr`` numbers a source within its own section: NumCalc's
    ``NC_ReadSoundSources`` reads ``numIncidentPlaneWaves_`` PLANE WAVES rows
    and ``numPointSources_`` POINT SOURCES rows as two independent lists, so
    the second entry is ``nr=1`` in its own section even if a plane wave came
    first in ``model.sources`` — a single index across the whole list would
    number the second source's section wrong.
    """
    plane_waves = [
        Source(nr=i + 1, vec=s.direction, real=s.amplitude.real,
              imag=s.amplitude.imag, curve_real=-1, curve_imag=-1)
        for i, s in enumerate(s for s in model.sources if isinstance(s, PlaneWave))
    ]
    point_sources = [
        Source(nr=i + 1, vec=s.position, real=s.amplitude.real,
              imag=s.amplitude.imag, curve_real=-1, curve_imag=-1)
        for i, s in enumerate(s for s in model.sources if isinstance(s, PointSource))
    ]
    return plane_waves, point_sources


def model_to_deck(model, *, n_eval_nodes, n_eval_elements, header=None,
                  title="model"):
    """Translate a solver-neutral Model into a NumCalc Deck.

    Args:
        model: the Model.
        n_eval_nodes / n_eval_elements: counts of the written EvalGrid.
            NumCalc's "Main Parameters I" only ever holds the *total* node/
            element count — the object/eval split never appears in NC.inp
            itself (the solver derives it from which mesh files it reads) —
            so the caller, who has already written (or sized) the EvalGrid,
            supplies these rather than the Deck deriving them.
        header: optional list of leading ``##`` comment lines for
            provenance. Defaults to none.
        title: free-text deck title (a comment in NC.inp, no semantic
            weight). Model carries no name of its own, so this defaults to a
            plain placeholder rather than pretending to read one.

    Returns:
        Deck, already checked with :meth:`Deck.validate`.
    """
    boundary, curves = _boundary_lines(model)
    plane_waves, point_sources = _sources(model)

    deck = Deck(
        title=title,
        frequencies=model.frequencies,
        speed_of_sound=model.medium.c,
        air_density=model.medium.rho,
        time_factor=model.time_factor,
        is_internal=0 if model.exterior else 1,
        n_total_nodes=model.mesh.n_nodes + n_eval_nodes,
        n_total_elements=model.mesh.n_elements + n_eval_elements,
        boundary=boundary,
        plane_waves=plane_waves,
        point_sources=point_sources,
        curves=curves,
        header=list(header) if header else [],
    )
    deck.validate()
    return deck


def _save_offsets(offsets, path):
    """Write the field-point offsets next to the bundle, as plain JSON.

    Metadata only — the offsets a solve *used* travel with the returned
    Results (see :meth:`Results.from_beout`); this file is what lets a bundle
    be inspected, or re-ingested, without re-running the solve.
    """
    with open(path, "w") as f:
        json.dump({name: idx.tolist() for name, idx in offsets.items()}, f)


class NumCalcSolver:
    """Run NumCalc on a Model.

    The working directory is the bundle: self-contained, relocatable, and
    resumable because bins are independent. With ``workdir=None`` a temporary
    one is used and the results are materialised into memory before cleanup —
    right for a quick comparison, wrong for a model too big to hold in RAM.
    """

    name = "numcalc"

    def solve(self, model, *, workdir=None, bins=None, jobs=1, options=()):
        """Solve ``model`` with NumCalc and return a :class:`Results`.

        Args:
            model: the Model to solve.
            workdir: bundle directory. Created if it does not exist. When
                None, a temporary directory is used and cleaned up before
                this call returns — the returned Results is fully
                materialised into memory first, so it survives the cleanup,
                but this is only appropriate for a model whose results fit
                comfortably in RAM.
            bins: 1-based frequency indices to compute (None = all). Taken
                literally: this is *the* set to compute, not a wish
                intersected with what is already present. A caller that
                wants a resume passes the missing bins (the CLI does).
            jobs: max concurrent NumCalc processes (see ``numcalckit.plan_jobs``).
            options: extra NumCalc argv flags for every job, as built by
                ``numcalckit.solver_flags`` (``-nitermax``, ``-check_normals``,
                ``-adapt_fmmlength``). NumCalc-specific and deliberately not
                part of the solver-neutral ``Solver`` protocol — a backend
                that is not NumCalc has nothing to do with these.

        Raises:
            ValueError: every field point was dropped by validation (or the
                model has none to begin with).
            RuntimeError: NumCalc exited non-zero for one or more jobs.
        """
        if workdir is not None:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            return self._solve_in(model, workdir, bins=bins, jobs=jobs,
                                  options=options)

        with tempfile.TemporaryDirectory(prefix="bemkit-numcalc-") as tmp:
            results = self._solve_in(model, Path(tmp), bins=bins, jobs=jobs,
                                     options=options)
            results.materialise()
        return results

    def export(self, model, workdir):
        """Write the NumCalc bundle for ``model`` into ``workdir``.

        Everything NumCalc reads — ObjectMesh/, EvalGrid/, NC.inp and the
        field-point offsets — and nothing it writes. Split out of
        :meth:`solve` because exporting is useful on its own: ``bem
        estimate-ram`` needs a bundle to point NumCalc's clustering pass at,
        and has no business solving anything to get one.

        Returns:
            ``{set name: (k,) int array}`` — the field-point offsets that
            survived validation, as :class:`Results` needs them.
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        raw_points, _ = fieldpoints.combine(model.fields)
        if len(raw_points) == 0:
            raise ValueError(
                "model has no field points to evaluate -- NumCalc needs at "
                "least one EvalGrid point (boundary-only results are not "
                "yet supported by this solver)")

        mask = fieldpoints.validate(raw_points, model.mesh,
                                    exterior=model.exterior)
        points, offsets = fieldpoints.combine(model.fields, mask)
        if len(points) == 0:
            raise ValueError(
                f"all {len(raw_points)} field point(s) were dropped by "
                f"validation (on the wrong side of the surface, or too "
                f"close to it) -- check the field-point sets against the "
                f"mesh and Model.exterior")

        write_object_mesh(model.mesh.vertices, model.mesh.triangles, workdir)
        write_eval_grid_nodes(points, workdir)
        n_eval_elements = write_eval_grid_elements(len(points), workdir)

        deck = model_to_deck(model, n_eval_nodes=len(points),
                             n_eval_elements=n_eval_elements)
        deck.write(workdir / "NC.inp")
        _save_offsets(offsets, workdir / "eval_offsets.json")
        return offsets

    def _solve_in(self, model, workdir, *, bins, jobs, options=()):
        offsets = self.export(model, workdir)

        nf = len(model.frequencies)
        jobs_to_run = plan_jobs([(workdir.name, workdir)], nf,
                                max_jobs=jobs, bins=bins, options=options)
        completed, failed = run_jobs(jobs_to_run, max_jobs=jobs)
        if failed:
            logs = ", ".join(str(j.nc_dir / j.log_name) for j in failed)
            raise RuntimeError(
                f"NumCalc failed for {len(failed)} of {len(jobs_to_run)} "
                f"job(s) in {workdir} -- see {logs}")

        return Results.from_beout(workdir, model.frequencies, offsets)

    def status(self, workdir):
        """Status of a run in ``workdir`` (see ``bemkit.solvers.base.Status``).

        The frequency count comes from the bundle's own ``NC.inp`` (via
        ``Deck.read``), not from a Model the caller may not have at hand --
        a workdir is meant to be inspectable on its own.

        Raises:
            FileNotFoundError: ``workdir`` has no ``NC.inp`` yet (nothing
                has been exported there), so "how many bins are missing"
                has no answer -- reporting a vacuous ``complete`` would be
                actively misleading.
        """
        workdir = Path(workdir)
        nc_inp = workdir / "NC.inp"
        if not nc_inp.exists():
            raise FileNotFoundError(
                f"{nc_inp} does not exist -- nothing has been exported to "
                f"{workdir} yet (call solve() first)")

        deck = Deck.read(nc_inp)
        nf = len(deck.frequencies)
        return Status(
            bins_present=tuple(available_bins(workdir)),
            bins_missing=tuple(missing_bins(workdir, nf)),
            running_pids=tuple(numcalc_pids(workdir)),
            warnings=tuple(solver_warnings(workdir)),
        )
