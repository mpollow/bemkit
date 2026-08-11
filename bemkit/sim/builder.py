"""Turn a :class:`~bemkit.config.ModelConfig` into a NumCalc :class:`Deck`.

This is the seam between bemkit's opinions and NumCalc's grammar. Everything
opinionated lives here:

* the frequency grid is the **FFT grid** ``sr``/``nbins`` — NumCalc itself
  accepts any list of frequencies (``docs/numcalc-feature-coverage.md`` §1.6);
* the mesh's *vibrating* physical group becomes a ``VELO`` boundary condition;
* geometric ``near=`` anchors have already been resolved to element ranges by
  :mod:`bemkit.sim.boundary`.

:mod:`bemkit.numcalc.deck` knows none of that and must not learn it.
"""

from datetime import datetime
from pathlib import Path

from bemkit.config import freq_grid
from bemkit.numcalc.deck import BC, Deck, Source, _contiguous_ranges

#: Fields 1-3 of a source row mean different things per section: a plane wave
#: travels along a *direction*, a point source sits at a *position*. Same
#: layout, so a row copied between the two sections is silently accepted and
#: silently wrong — the error message names the right triple for the section.
_SOURCE_TRIPLE = {
    "PLANE WAVES": ("dx, dy, dz",
                    "dx, dy, dz is the direction the wave travels in, not a "
                    "position (NumCalc normalises it)"),
    "POINT SOURCES": ("x0, y0, z0",
                      "x0, y0, z0 is the source position in meters"),
}


def _count(path):
    """First line of a NumCalc Nodes.txt/Elements.txt file is its count."""
    with open(path) as f:
        return int(f.readline().strip())


def _sources(cfg, section):
    """Validate a PLANE WAVES / POINT SOURCES table into Source records.

    Every entry must be a list of 8 values (matching the NC.inp line layout);
    anything else raises, because a dropped entry would leave the header count
    claiming a source line that never appears and NumCalc would misparse.
    """
    entries = cfg.get(section) or {}
    triple, meaning = _SOURCE_TRIPLE.get(section, ("x, y, z", ""))
    out = []
    for key, val in entries.items():
        if not (isinstance(val, list) and len(val) == 8):
            raise ValueError(
                f'["{section}"] entry {key!r} must be a list of 8 values '
                f"[nr, {triple}, ampl_real, -1, ampl_imag, -1], "
                f"got: {val!r}\n  {meaning}.")
        # Curve refs (positions 5/7) are always -1: frequency-dependent
        # amplitude CURVES are not wired up to the config surface yet.
        out.append(Source(nr=val[0], vec=tuple(val[1:4]),
                          real=val[4], imag=val[6],
                          curve_real=-1, curve_imag=-1))
    return out


def _header(config, variant, nc_dir, counts, n_frequencies):
    """The leading ``##`` comment block — provenance, not semantics."""
    n_obj_n, n_obj_e, n_eval_n, n_eval_e = counts
    offsets_file = Path(nc_dir) / "eval_offsets.json"
    if offsets_file.exists():
        from bemkit.sim.fieldpoints import load_offsets
        sets = load_offsets(offsets_file)["sets"]
        summary = ", ".join(f"{n} ({i['n_valid']})" for n, i in sets.items())
    else:
        summary = f"{n_eval_n} points"
    rule = "##" + "-" * 43
    return [
        rule,
        "## Created using bemkit",
        f"## Date: {datetime.today().strftime('%Y-%m-%d')}",
        f"## Variant: {variant}",
        f"## Sample rate: {config.get('sr')} Hz | Frequencies: {n_frequencies}",
        f"## Mesh: {n_obj_n} nodes, {n_obj_e} elements | "
        f"EvalGrid: {n_eval_n} nodes, {n_eval_e} elements",
        f"## Fieldpoints: {summary}",
        rule,
    ]


def deck_from_config(config, variant, nc_dir, *, vibrating_indices=None,
                     boundary_ranges=None, verbose=True):
    """Build the NumCalc deck for one variant.

    ``ObjectMesh/`` and ``EvalGrid/`` must already be written into ``nc_dir`` —
    their counts are read back from disk, which is also what bounds-checks the
    boundary element ranges.

    Args:
        config: ModelConfig (or a raw config.toml dict).
        variant: variant name. A ``[BOUNDARY.<variant>]`` subsection, when
            present, is used EXCLUSIVELY; otherwise the global entries apply.
        nc_dir: the ``numcalc/{variant}/`` bundle directory.
        vibrating_indices: element indices driven by the mesh's vibrating
            physical group; ``[BOUNDARY]`` entries are appended additionally.
        boundary_ranges: pre-resolved ``(start, end, velocity)`` tuples, used
            instead of parsing ``config[BOUNDARY]`` here. ``export_variant``
            passes these because it has resolved geometric ``near=`` anchors
            against the mesh; when None the config table is parsed here and
            only literal forms are accepted.

    Returns:
        Deck
    """
    nc_dir = Path(nc_dir)
    counts = (_count(nc_dir / "ObjectMesh" / "Nodes.txt"),
              _count(nc_dir / "ObjectMesh" / "Elements.txt"),
              _count(nc_dir / "EvalGrid" / "Nodes.txt"),
              _count(nc_dir / "EvalGrid" / "Elements.txt"))
    n_obj_nodes, n_obj_elements, n_eval_nodes, n_eval_elements = counts

    # ModelConfig supports dict access, but a raw dict has no typed defaults —
    # fill them from the dataclass so both callers behave identically.
    v = dict(config.raw) if hasattr(config, "raw") else dict(config)
    if hasattr(config, "__dataclass_fields__"):
        for name in ("modelname", "sr", "nbins", "bem_type", "air_density",
                     "speed_of_sound", "time_factor", "is_internal"):
            v.setdefault(name, getattr(config, name))
    v.setdefault("time_factor", 1.0)
    v.setdefault("is_internal", 0)
    v.setdefault("bem_type", 4)
    v.setdefault("air_density", 1.2)
    v.setdefault("speed_of_sound", 340.0)

    n_frequencies = v["nbins"] - 1
    frequencies = freq_grid(v["sr"], v["nbins"])

    sym = v.get("SYMMETRY") or {}
    planes = list(sym.get("planes", [])) or None

    # -- boundary conditions --
    velocity = v.get("velocity", 1.0)
    # Materialise to a list first: vibrating_indices arrives as a numpy array
    # from the mesh's physical groups, and `array or []` raises.
    vib = [] if vibrating_indices is None else list(vibrating_indices)
    elems = [(s, e, velocity) for s, e in _contiguous_ranges(vib)]
    had_boundary_table = "BOUNDARY" in v
    if boundary_ranges is not None:
        elems += list(boundary_ranges)
    elif had_boundary_table:
        # Same parser as export_variant, so variant matching (including the
        # __recipe/@profile stripping) cannot diverge between the two paths.
        # Geometric near= anchors need a mesh and raise here — resolve them
        # via export_variant / resolve_boundary instead.
        from bemkit.sim.boundary import flatten, resolve_boundary
        elems += flatten(resolve_boundary(
            v.get("BOUNDARY"), variant, mesh_file=None,
            default_velocity=velocity))

    boundary = []
    for s, e, v0 in elems:
        if not (0 <= int(s) <= int(e) < n_obj_elements):
            raise ValueError(
                f"BOUNDARY element range [{s}, {e}] for variant '{variant}' is "
                f"out of bounds: the mesh has {n_obj_elements} elements (valid "
                f"indices 0..{n_obj_elements - 1}). A stale index usually means "
                f"the BC was picked on a different mesh — re-pick or use "
                f"geometric 'near=' anchors.")
        boundary.append(BC(start=int(s), end=int(e), kind="VELO",
                           real=float(v0), imag=0.0,
                           curve_real=-1, curve_imag=-1))

    if not boundary and (had_boundary_table or vib):
        print(f"  WARNING: variant '{variant}' has a BOUNDARY definition but no "
              f"vibrating elements were emitted — the surface will be fully rigid. "
              f"Check that [BOUNDARY.{variant}] exists and its entries are well-formed.")

    if verbose:
        print(f"Mesh: {n_obj_nodes} nodes, {n_obj_elements} elements | "
              f"EvalGrid: {n_eval_nodes} nodes")

    return Deck(
        title=v["modelname"],
        frequencies=frequencies,
        speed_of_sound=v["speed_of_sound"],
        air_density=v["air_density"],
        time_factor=v["time_factor"],
        bem_type=v["bem_type"],
        is_internal=v["is_internal"],
        n_total_nodes=n_obj_nodes + n_eval_nodes,
        n_total_elements=n_obj_elements + n_eval_elements,
        symmetry_planes=planes,
        symmetry_coordinates=(list(sym.get("coordinates", [0.0, 0.0, 0.0]))
                              if planes else None),
        boundary=boundary,
        plane_waves=_sources(v, "PLANE WAVES"),
        point_sources=_sources(v, "POINT SOURCES"),
        header=_header(config, variant, nc_dir, counts, n_frequencies),
    )
