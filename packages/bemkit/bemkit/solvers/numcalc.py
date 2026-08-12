"""Model -> NumCalc Deck.

This is the single place NumCalc-isms are allowed to live. The model layer
above (:mod:`bemkit.model`) knows nothing of decks, element ranges or CURVES;
:mod:`pynumcalc` below knows nothing of models. Everything opinionated about
*how* a solver-neutral :class:`~bemkit.model.Model` becomes an ``NC.inp`` deck
belongs here.
"""

from pynumcalc.deck import BC as DeckBC
from pynumcalc.deck import Curve, Deck, Source, _contiguous_ranges

from bemkit.model.sources import PlaneWave, PointSource

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
