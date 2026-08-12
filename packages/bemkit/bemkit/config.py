"""Configuration for NumCalc BEM simulations.

ModelConfig is the single configuration object of the toolkit. It also
supports dict-style access (``config["sr"]``, ``config.get(...)``) so code
paths that historically took the raw config.toml dict keep working.

config.toml layout: solver settings live in the required ``[bem]`` section;
``[geometry]``/``[fieldpoints]`` declare their stage inputs explicitly
(``exec`` scripts, declarative generator sets, or files); ``[mesh]`` holds
shared gmsh parameters; ``[variants]`` names variants with optional
per-variant mesh overrides.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# [bem] keys and their defaults (None = required).
_BEM_KEYS = {
    "sr": None,
    "nbins": None,
    "air_density": 1.2,
    "speed_of_sound": 340.0,
    "bem_type": 4,
    "time_factor": -1.0,
    "is_internal": 0,
    # -- solver knobs; every default is NumCalc's own, so omitting them all
    #    reproduces the deck bemkit wrote before they existed --
    "solver": 0,               # ISOLVER: 0 iterative (CGS), 4 direct (Gauss)
    "min_expansion_terms": 0,  # 0 = NumCalc's default of 8; else 5..10
    "cluster_edge_length": 0.0,  # ClusEdgL0 in m; 0 = NumCalc's own default
    "preconditioner": 0,       # IPRECOND in [-3, 2]
}

#: Deck fields the raw ``[nc]`` escape hatch may set. Deliberately only the
#: scalars: everything else on the deck is *derived* (frequencies from
#: sr/nbins, boundary from [BOUNDARY], counts from the mesh), and letting a
#: raw table shadow those would make the config lie about what was exported.
_NC_OVERRIDE_FIELDS = (
    "speed_of_sound", "air_density", "time_factor",
    "bem_type", "is_internal", "n_element_groups",
    "min_expansion_terms", "isolver", "cluster_edge_length", "preconditioner",
)

#: Derived deck fields, named so the error can say where they really come from.
_NC_DERIVED = {
    "frequencies": "[bem] sr/nbins",
    "boundary": "[BOUNDARY]",
    "plane_waves": '["PLANE WAVES"]',
    "point_sources": '["POINT SOURCES"]',
    "symmetry_planes": "[SYMMETRY]",
    "symmetry_coordinates": "[SYMMETRY]",
    "title": "modelname",
    "n_total_nodes": "the exported mesh",
    "n_total_elements": "the exported mesh",
}

_MESH_KEYS = ("lc_min", "lc_max", "dist_min", "dist_max", "distance")

#: [groups] keys — map a bemkit BC role onto this mesh's physical-group name.
#: Only ``vibrating`` is functional: every element not in it is rigid by
#: default, so naming the rigid group would carry no information.
_GROUP_KEYS = ("vibrating",)

#: How the size ramp measures distance from the vibrating surface.
#: "geodesic" travels over the surface and is STEP/CAD only.
_DISTANCE_MODES = ("euclidean", "geodesic")

#: Declarative fieldpoint set types, dispatching to make_fields_<type>.
FIELDPOINT_GENERATORS = ("onaxis", "polar", "plane")


def _check_exec(value, where):
    if isinstance(value, str):
        return
    if isinstance(value, list) and value and all(isinstance(s, str) for s in value):
        return
    raise ValueError(f"{where}: exec must be a script path or a list of "
                     f"script paths, got {value!r}")


def _check_mesh_params(tbl, where):
    unknown = [k for k in tbl if k not in _MESH_KEYS]
    if unknown:
        raise ValueError(f"{where}: unknown mesh key(s): {', '.join(unknown)} "
                         f"(known: {', '.join(_MESH_KEYS)})")
    mode = tbl.get("distance")
    if mode is not None and mode not in _DISTANCE_MODES:
        raise ValueError(f"{where}: distance must be one of "
                         f"{', '.join(_DISTANCE_MODES)}, got {mode!r}")


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_point(x):
    return isinstance(x, list) and len(x) == 3 and all(_is_num(c) for c in x)


#: Boundary-condition types bemkit exposes. TRAD/TRIM are deliberately absent:
#: NumCalc parses them and then hard-errors ("not supported").
BC_TYPES = ("velo", "pres", "admi", "impe")

#: Above this, an admittance or impedance is almost certainly in SI units
#: rather than normalized by rho*c. Normalized values sit around 0-2 for
#: admittance and 0.5-20 for impedance; an SI impedance for air starts near 400.
_BC_IMPLAUSIBLE = 100.0


def _check_bc_type(key, val, where):
    """Validate the ``type``/``value`` pair of a geometric [BOUNDARY] entry."""
    if "type" not in val and "value" not in val:
        return
    kind = str(val.get("type", "velo")).lower()
    if kind not in BC_TYPES:
        extra = ("  TRAD/TRIM are parsed by NumCalc and then hard-error, so "
                 "they are not offered.") if kind in ("trad", "trim") else ""
        raise ValueError(
            f"{where}: entry '{key}': type must be one of "
            f"{', '.join(BC_TYPES)}, got {val['type']!r}.{extra}")
    if "velocity" in val or "v" in val:
        raise ValueError(
            f"{where}: entry '{key}': give either velocity/v (shorthand for a "
            f"VELO condition) or type/value, not both")

    value = val.get("value", 0.0 if kind == "pres" else 1.0)
    if _is_num(value):
        re_, im_ = float(value), 0.0
    elif (isinstance(value, list) and len(value) == 2
            and all(_is_num(x) for x in value)):
        re_, im_ = float(value[0]), float(value[1])
    else:
        raise ValueError(
            f"{where}: entry '{key}': value must be a number or [real, imag], "
            f"got {value!r}")

    if kind in ("admi", "impe") and max(abs(re_), abs(im_)) > _BC_IMPLAUSIBLE:
        raise ValueError(
            f"{where}: entry '{key}': {kind.upper()} value {value!r} looks "
            f"like SI units. NumCalc wants a **normalized** quantity — "
            f"admittance divided by rho0*c (so 1.0 is a matched, anechoic "
            f"termination), impedance multiplied by the same. Divide an SI "
            f"admittance by ~1/408, or an SI impedance by ~408.")


def _check_bc_curve(key, val, where):
    """A ``curve = "<name>"`` reference is ADMI-only.

    NumCalc ignores curves for PRES/VELO, and ``IMPE`` combined with a curve is
    a hard ``exit(-1)`` whose message says to use ADMI instead
    (``NC_Input.cpp:1275``). Catching it here beats losing a run to it.
    """
    if "curve" not in val:
        return
    if not isinstance(val["curve"], str):
        raise ValueError(f"{where}: entry '{key}': curve must be the name of a "
                         f"[curves.<name>] table, got {val['curve']!r}")
    kind = str(val.get("type", "velo")).lower()
    if kind != "admi":
        why = ("NumCalc exits(-1) on IMPE with a curve and tells you to use "
               "ADMI" if kind == "impe" else
               f"NumCalc ignores curves for {kind.upper()}")
        raise ValueError(
            f"{where}: entry '{key}': a frequency curve is only supported for "
            f'type = "admi" ({why}).')


def _check_frequencies(tbl, path):
    """Validate the optional [frequencies] table.

    NumCalc's frequency grid is a curve, not a range (see
    ``docs/numcalc-feature-coverage.md`` §1.6), so any ascending list of
    frequencies is solvable. ``sr``/``nbins`` remains the default because
    impulse responses need a uniform FFT grid; this is the opt-out.
    """
    where = f"{path}: [frequencies]"
    if not isinstance(tbl, dict):
        raise ValueError(f"{where} must be a table")
    unknown = [k for k in tbl if k not in ("list", "log")]
    if unknown:
        raise ValueError(f"{where}: unknown key(s): {', '.join(unknown)} "
                         f"(known: list, log)")
    given = [k for k in ("list", "log") if k in tbl]
    if len(given) != 1:
        raise ValueError(f"{where}: needs exactly one of list/log, "
                         f"got {given or 'neither'}")

    if "list" in tbl:
        vals = tbl["list"]
        if not (isinstance(vals, list) and len(vals) >= 1
                and all(_is_num(f) and f > 0 for f in vals)):
            raise ValueError(f"{where}: list must be positive numbers (Hz), "
                             f"got {vals!r}")
        if list(vals) != sorted(vals) or len(set(vals)) != len(vals):
            raise ValueError(f"{where}: list must be strictly ascending")
        return

    log = tbl["log"]
    if not isinstance(log, dict):
        raise ValueError(f"{where}: log must be a table with start/stop/count")
    unknown = [k for k in log if k not in ("start", "stop", "count")]
    if unknown:
        raise ValueError(f"{where}: log: unknown key(s): {', '.join(unknown)}")
    for k in ("start", "stop", "count"):
        if k not in log:
            raise ValueError(f"{where}: log is missing '{k}'")
    if not (_is_num(log["start"]) and log["start"] > 0):
        raise ValueError(f"{where}: log.start must be > 0 Hz")
    if not (_is_num(log["stop"]) and log["stop"] > log["start"]):
        raise ValueError(f"{where}: log.stop must be greater than log.start")
    if not (isinstance(log["count"], int) and not isinstance(log["count"], bool)
            and log["count"] >= 2):
        raise ValueError(f"{where}: log.count must be an integer >= 2")


def _check_curves(tbl, path):
    """Validate the [curves] table: named frequency-dependent BC values."""
    where = f"{path}: [curves]"
    if not isinstance(tbl, dict):
        raise ValueError(f"{where} must be a table")
    for name, spec in tbl.items():
        sub = f"{path}: [curves.{name}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{sub}: must be a table with frequencies/real"
                             f"[/imag] arrays")
        unknown = [k for k in spec if k not in ("frequencies", "real", "imag")]
        if unknown:
            raise ValueError(f"{sub}: unknown key(s): {', '.join(unknown)} "
                             f"(known: frequencies, real, imag)")
        for req in ("frequencies", "real"):
            if req not in spec:
                raise ValueError(f"{sub}: missing required key '{req}'")
        freqs = spec["frequencies"]
        if not (isinstance(freqs, list) and len(freqs) >= 2
                and all(_is_num(f) for f in freqs)):
            raise ValueError(f"{sub}: frequencies must be a list of at least "
                             f"two numbers (Hz), got {freqs!r}")
        if list(freqs) != sorted(freqs):
            raise ValueError(f"{sub}: frequencies must be ascending — NumCalc "
                             f"interpolates in order and does not sort")
        for part in ("real", "imag"):
            if part not in spec:
                continue
            vals = spec[part]
            if not (isinstance(vals, list) and all(_is_num(x) for x in vals)):
                raise ValueError(f"{sub}: {part} must be a list of numbers, "
                                 f"got {vals!r}")
            if len(vals) != len(freqs):
                raise ValueError(
                    f"{sub}: {part} has {len(vals)} values but frequencies has "
                    f"{len(freqs)} — they index the same points")
            if max((abs(x) for x in vals), default=0.0) > _BC_IMPLAUSIBLE:
                raise ValueError(
                    f"{sub}: {part} value looks like SI units. A CURVES value "
                    f"scales a **normalized** admittance (Y / (rho0*c)); 1.0 is "
                    f"a matched, anechoic termination.")


def _check_bc_entry(key, val, where):
    """Validate one [BOUNDARY] entry: [s,e,v], [[s,e,v],...], or {near=...}.

    Shared grammar for config-load validation and resolve_boundary, so a
    library caller bypassing config.toml hits the same loud failures.
    """
    if isinstance(val, dict):
        unknown = [k for k in val
                   if k not in ("near", "radius", "velocity", "v",
                                "type", "value", "curve")]
        if unknown:
            raise ValueError(
                f"{where}: entry '{key}': unknown key(s) {', '.join(unknown)} "
                f"(known: near, radius, velocity, v, type, value, curve)")
        _check_bc_type(key, val, where)
        _check_bc_curve(key, val, where)
        near = val.get("near")
        if near is None:
            raise ValueError(f"{where}: entry '{key}': geometric entries need "
                             f"near = [x, y, z] (meters)")
        if not (_is_point(near)
                or (isinstance(near, list) and near
                    and all(_is_point(p) for p in near))):
            raise ValueError(f"{where}: entry '{key}': near must be [x, y, z] "
                             f"or a list of such points, got {near!r}")
        if "radius" in val and not (_is_num(val["radius"]) and val["radius"] > 0):
            raise ValueError(f"{where}: entry '{key}': radius must be a "
                             f"positive number (meters), got {val['radius']!r}")
        for vk in ("velocity", "v"):
            if vk in val and not _is_num(val[vk]):
                raise ValueError(f"{where}: entry '{key}': {vk} must be a "
                                 f"number, got {val[vk]!r}")
        return
    if isinstance(val, list):
        rows = val if (val and isinstance(val[0], list)) else [val]
        for r in rows:
            if not (isinstance(r, list) and len(r) == 3
                    and all(_is_num(x) for x in r)
                    and float(r[0]).is_integer() and float(r[1]).is_integer()):
                raise ValueError(
                    f"{where}: entry '{key}': expected [start, end, velocity] "
                    f"(two element indices and a velocity), got {r!r}")
        return
    raise ValueError(f"{where}: entry '{key}': must be [start, end, velocity], "
                     f"a list of such ranges, or {{ near = ... }}, got {val!r}")


def _check_boundary(bc, path):
    """Validate a [BOUNDARY] table: entries, or one level of per-variant
    subtables (a dict without a ``near`` key) holding entries."""
    where = f"{path}: [BOUNDARY]"
    if not isinstance(bc, dict):
        raise ValueError(f"{where} must be a table")
    for key, val in bc.items():
        if isinstance(val, dict) and "near" not in val:
            sub = f"{path}: [BOUNDARY.{key}]"
            for k2, v2 in val.items():
                if isinstance(v2, dict) and "near" not in v2:
                    raise ValueError(
                        f"{sub}: entry '{k2}' looks like a nested subtable — "
                        f"only one level of [BOUNDARY.<variant>] is supported")
                _check_bc_entry(k2, v2, sub)
        else:
            _check_bc_entry(key, val, where)


def _check_symmetry(sym, path):
    where = f"{path}: [SYMMETRY]"
    if not isinstance(sym, dict):
        raise ValueError(f"{where} must be a table")
    unknown = [k for k in sym if k not in ("planes", "coordinates")]
    if unknown:
        raise ValueError(f"{where}: unknown key(s): {', '.join(unknown)} "
                         f"(known: planes, coordinates)")
    planes = sym.get("planes")
    if planes is not None and not (
            isinstance(planes, list) and len(planes) == 3
            and all(isinstance(p, int) and not isinstance(p, bool)
                    for p in planes)):
        raise ValueError(f"{where}: planes must be three integers "
                         f"(e.g. [1, 0, 0]), got {planes!r}")
    coords = sym.get("coordinates")
    if coords is not None and not _is_point(coords):
        raise ValueError(f"{where}: coordinates must be [x, y, z] numbers, "
                         f"got {coords!r}")


def _check_sources(tbl, section, path):
    """Validate a ["PLANE WAVES"] / ["POINT SOURCES"] table structurally.

    Each entry is one 8-number row or a list of rows; the semantic checks
    (direction not all-zero, curve refs) stay in sim.writers at export time.
    """
    where = f"{path}: [{section}]"
    if not isinstance(tbl, dict):
        raise ValueError(f"{where} must be a table")
    for key, val in tbl.items():
        rows = (val if (isinstance(val, list) and val
                        and isinstance(val[0], list)) else [val])
        for r in rows:
            if not (isinstance(r, list) and len(r) == 8
                    and all(_is_num(x) for x in r)):
                raise ValueError(
                    f"{where}: entry '{key}': each row needs 8 numbers "
                    f"[nr, a, b, c, ampl_re, -1, ampl_im, -1], got {r!r}")


def _check_solver_keys(bem, path):
    """Range-check the [bem] solver knobs against what NumCalc accepts.

    NumCalc hard-errors on most of these, but two of them fail *quietly*
    instead, which is why they are checked here rather than left to the solver:
    a direct solve is silently downgraded to iterative whenever FMM is on
    (``NC_Input.cpp:290``), and the preconditioner is silently forced to 2 when
    the direct solver runs (``:323``).
    """
    where = f"{path}: [bem]"
    bem_type = bem.get("bem_type", 4)
    solver = bem.get("solver", 0)
    if solver not in (0, 4):
        raise ValueError(f"{where}: solver (ISOLVER) must be 0 (iterative CGS) "
                         f"or 4 (direct Gauss), got {solver!r}")
    if solver == 4 and bem_type != 0:
        raise ValueError(
            f"{where}: solver = 4 (direct) only applies with bem_type = 0 "
            f"(TBEM); with bem_type = {bem_type} NumCalc silently falls back to "
            f"the iterative solver, so the setting would do nothing. Set "
            f"bem_type = 0 as well, or drop solver.")

    met = bem.get("min_expansion_terms", 0)
    if not (isinstance(met, int) and not isinstance(met, bool)
            and (met == 0 or 5 <= met <= 10)):
        raise ValueError(
            f"{where}: min_expansion_terms must be 0 (NumCalc's default of 8) "
            f"or an integer in [5, 10], got {met!r}")

    cel = bem.get("cluster_edge_length", 0.0)
    if not (_is_num(cel) and cel >= 0):
        raise ValueError(f"{where}: cluster_edge_length must be a "
                         f"non-negative number (meters, 0 = NumCalc's "
                         f"default), got {cel!r}")

    pre = bem.get("preconditioner", 0)
    if not (isinstance(pre, int) and not isinstance(pre, bool)
            and -3 <= pre <= 2):
        raise ValueError(f"{where}: preconditioner (IPRECOND) must be an "
                         f"integer in [-3, 2], got {pre!r}")
    if solver == 4 and pre != 0:
        raise ValueError(
            f"{where}: preconditioner is ignored when solver = 4 — NumCalc "
            f"forces it to 2 for the direct solver. Drop one of the two.")


def _check_nc(tbl, path):
    """Validate the raw [nc] deck-override table."""
    where = f"{path}: [nc]"
    if not isinstance(tbl, dict):
        raise ValueError(f"{where} must be a table")
    for key, val in tbl.items():
        if key in _NC_DERIVED:
            raise ValueError(
                f"{where}: '{key}' is derived from {_NC_DERIVED[key]} and "
                f"cannot be overridden here — set it there instead, so the "
                f"config keeps describing what was exported")
        if key not in _NC_OVERRIDE_FIELDS:
            raise ValueError(
                f"{where}: unknown deck field '{key}' "
                f"(settable: {', '.join(_NC_OVERRIDE_FIELDS)})")
        if not _is_num(val):
            raise ValueError(f"{where}: '{key}' must be a number, got {val!r}")


def _validate(raw, path):
    """Validate config.toml structure; raise ValueError with migration hints."""
    old = [k for k in _BEM_KEYS if k in raw]
    if old:
        raise ValueError(
            f"{path}: top-level key(s) {', '.join(old)} have moved into the "
            f"[bem] section (config format change — see HOWTO.md)")
    bem = raw.get("bem")
    if not isinstance(bem, dict):
        raise ValueError(f"{path} is missing the required [bem] section "
                         f"(sr, nbins, and other solver settings)")
    missing = [k for k in ("sr", "nbins") if k not in bem]
    if len(missing) == 1:
        # Half a grid is always a mistake, [frequencies] table or not: the two
        # keys only mean anything together.
        raise ValueError(
            f"{path}: [bem] has {'nbins' if 'sr' in missing else 'sr'} but not "
            f"{missing[0]} — give both (the uniform FFT grid) or neither "
            f"(with a [frequencies] table instead)")
    if missing and "frequencies" not in raw:
        raise ValueError(f"{path}: [bem] is missing required key(s): "
                         f"{', '.join(missing)}")
    unknown = [k for k in bem if k not in _BEM_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown [bem] key(s): {', '.join(unknown)} "
                         f"(known: {', '.join(_BEM_KEYS)})")
    _check_solver_keys(bem, path)

    geo = raw.get("geometry", {})
    if not isinstance(geo, dict):
        raise ValueError(f"{path}: [geometry] must be a table")
    unknown = [k for k in geo if k != "exec"]
    if unknown:
        raise ValueError(f"{path}: unknown [geometry] key(s): "
                         f"{', '.join(unknown)} (known: exec)")
    if "exec" in geo:
        _check_exec(geo["exec"], f"{path}: [geometry]")

    mesh = raw.get("mesh", {})
    if not isinstance(mesh, dict):
        raise ValueError(f"{path}: [mesh] must be a table")
    _check_mesh_params(mesh, f"{path}: [mesh]")

    groups = raw.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"{path}: [groups] must be a table")
    unknown = [k for k in groups if k not in _GROUP_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown [groups] key(s): {', '.join(unknown)} "
                         f"(known: {', '.join(_GROUP_KEYS)})")
    for k, v in groups.items():
        if not isinstance(v, str):
            raise ValueError(f"{path}: [groups] {k} must be a physical-group "
                             f"name (string), got {v!r}")

    fp = raw.get("fieldpoints", {})
    if isinstance(fp, str):
        raise ValueError(
            f"{path}: fieldpoints = \"{fp}\" is the old directory form — "
            f"delete the line (fieldpoints/*.npy files are found "
            f"automatically) or define [fieldpoints.<name>] set tables")
    if not isinstance(fp, dict):
        raise ValueError(f"{path}: [fieldpoints] must be a table")
    for name, spec in fp.items():
        where = f"{path}: [fieldpoints.{name}]"
        if name == "exec":
            _check_exec(spec, f"{path}: [fieldpoints]")
            continue
        if isinstance(spec, str):
            raise ValueError(
                f"{path}: fieldpoints entry {name} = \"{spec}\" is the old "
                f"path form — use [fieldpoints.{name}] with file = \"{spec}\"")
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: must be a table with type = ... or "
                             f"file = ...")
        keys = [k for k in ("type", "file") if k in spec]
        if len(keys) != 1:
            raise ValueError(f"{where}: needs exactly one of type/file, "
                             f"got {keys or 'neither'}")
        if "type" in spec and spec["type"] not in FIELDPOINT_GENERATORS:
            raise ValueError(
                f"{where}: unknown type {spec['type']!r} "
                f"(known: {', '.join(FIELDPOINT_GENERATORS)})")
        if "file" in spec and len(spec) > 1:
            extra = [k for k in spec if k != "file"]
            raise ValueError(f"{where}: file sets take no other keys "
                             f"(got: {', '.join(extra)})")

    variants = raw.get("variants", {})
    if not isinstance(variants, dict):
        raise ValueError(f"{path}: [variants] must be a table")
    for name, v in variants.items():
        where = f"{path}: [variants.{name}]"
        if not isinstance(v, dict):
            raise ValueError(f"{where}: must be a table")
        unknown = [k for k in v if k not in ("mesh", "source")]
        if unknown:
            raise ValueError(f"{where}: unknown key(s): {', '.join(unknown)} "
                             f"(known: mesh, source)")
        m = v.get("mesh")
        if m is not None and not isinstance(m, (str, dict)):
            raise ValueError(f"{where}: mesh must be a mesh-file stem (string)"
                             f" or a table of mesh-parameter overrides")
        if isinstance(m, dict):
            _check_mesh_params(m, f"{where}: mesh")
        if "source" in v and not isinstance(v["source"], str):
            raise ValueError(f"{where}: source must be a geometry/ file stem")

    # Physics sections: a silent mistake here yields plausible-but-wrong
    # results (a dropped BC exports a rigid surface that solves to silence),
    # so they get the same typo guard as the solver/mesh sections.
    if "nc" in raw:
        _check_nc(raw["nc"], path)
    if "curves" in raw:
        _check_curves(raw["curves"], path)
    if "frequencies" in raw:
        _check_frequencies(raw["frequencies"], path)
    if "BOUNDARY" in raw:
        _check_boundary(raw["BOUNDARY"], path)
    if "SYMMETRY" in raw:
        _check_symmetry(raw["SYMMETRY"], path)
    for section in ("PLANE WAVES", "POINT SOURCES"):
        if section in raw:
            _check_sources(raw[section], section, path)
    if "velocity" in raw and not _is_num(raw["velocity"]):
        raise ValueError(f"{path}: top-level velocity must be a number "
                         f"(default velocity for BOUNDARY ranges), "
                         f"got {raw['velocity']!r}")


@dataclass(frozen=True)
class ModelConfig:
    """Typed wrapper around a model directory's config.toml."""

    model_dir: Path
    modelname: str
    sr: int | None = None
    nbins: int | None = None
    air_density: float = 1.2
    speed_of_sound: float = 340.0
    bem_type: int = 4
    time_factor: float = -1.0
    is_internal: int = 0
    solver: int = 0
    min_expansion_terms: int = 0
    cluster_edge_length: float = 0.0
    preconditioner: int = 0
    mesh_cfg: dict = field(default_factory=dict)
    geometry_cfg: dict = field(default_factory=dict)
    fieldpoints_cfg: dict = field(default_factory=dict)
    groups_cfg: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw, model_dir, source=None):
        """Build a validated ModelConfig from an in-memory dict.

        The seam for driving bemkit programmatically from another package:
        identical validation to config.toml loading, no file required.
        ``source`` names the origin in error messages.
        """
        model_dir = Path(model_dir).resolve()
        _validate(raw, source or f"<config for {model_dir.name}>")
        bem = raw["bem"]
        raw.setdefault("modelname", model_dir.name)
        return cls(
            model_dir=model_dir,
            modelname=raw["modelname"],
            sr=bem.get("sr"),
            nbins=bem.get("nbins"),
            air_density=bem.get("air_density", 1.2),
            speed_of_sound=bem.get("speed_of_sound", 340.0),
            bem_type=bem.get("bem_type", 4),
            time_factor=bem.get("time_factor", -1.0),
            is_internal=bem.get("is_internal", 0),
            solver=bem.get("solver", 0),
            min_expansion_terms=bem.get("min_expansion_terms", 0),
            cluster_edge_length=bem.get("cluster_edge_length", 0.0),
            preconditioner=bem.get("preconditioner", 0),
            mesh_cfg=raw.get("mesh", {}),
            geometry_cfg=raw.get("geometry", {}),
            fieldpoints_cfg=raw.get("fieldpoints", {}),
            groups_cfg=raw.get("groups", {}),
            raw=raw,
        )

    @classmethod
    def from_dir(cls, model_dir):
        """Load config.toml; modelname defaults to the directory name."""
        model_dir = Path(model_dir).resolve()
        cfg_path = model_dir / "config.toml"
        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)
        return cls.from_dict(raw, model_dir, source=str(cfg_path))

    # -- dict-style access (backward compatible with raw-dict callers) --
    #
    # Lookups consult config.toml first, then fall back to the typed dataclass
    # fields, so config.get("air_density") returns the effective value even
    # though the toml key lives inside [bem] — dict access never diverges
    # from attribute access.

    def __getitem__(self, key):
        if key in self.raw:
            return self.raw[key]
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.raw or key in self.__dataclass_fields__

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # -- helpers --

    def freq_array(self):
        """The frequencies this model solves, in Hz.

        ``sr``/``nbins`` gives the uniform FFT grid (DC excluded). A
        ``[frequencies]`` table replaces it with an arbitrary ascending list —
        NumCalc solves any of them (§1.6), but time-domain results need the
        uniform one.
        """
        spec = self.raw.get("frequencies")
        if spec:
            if "list" in spec:
                return np.asarray(spec["list"], dtype=float)
            log = spec["log"]
            return np.logspace(np.log10(log["start"]), np.log10(log["stop"]),
                               log["count"])
        return freq_grid(self.sr, self.nbins)

    @property
    def n_frequencies(self):
        """How many frequency bins a full run computes."""
        spec = self.raw.get("frequencies")
        if spec:
            return len(spec["list"]) if "list" in spec else spec["log"]["count"]
        return self.nbins - 1

    @property
    def is_uniform_grid(self):
        """True when the frequencies form the FFT grid an IR needs."""
        return not self.raw.get("frequencies")

    def freq_bin(self, freq_hz):
        """(0-based index, 1-based bin, actual Hz) of the nearest frequency."""
        freqs = self.freq_array()
        idx = int(np.argmin(np.abs(freqs - freq_hz)))
        return idx, idx + 1, float(freqs[idx])

    def resolve_bins(self, *, freq=None, bins=None, preview=None):
        """Resolve a partial-band selection against this model's grid."""
        return resolve_bins(self.freq_array(), freq=freq, bins=bins,
                            preview=preview)

    @property
    def boundary_cfg(self):
        """Validated [BOUNDARY] table ({} when absent)."""
        return self.raw.get("BOUNDARY", {})

    @property
    def symmetry_cfg(self):
        """Validated [SYMMETRY] table ({} when absent)."""
        return self.raw.get("SYMMETRY", {})

    @property
    def sources_cfg(self):
        """Validated incident-field tables, keyed by NC.inp section name."""
        return {s: self.raw.get(s, {})
                for s in ("PLANE WAVES", "POINT SOURCES")}

    @property
    def default_velocity(self):
        """Top-level ``velocity`` key: default velocity for BC ranges (1.0)."""
        return self.raw.get("velocity", 1.0)

    def vibrating_group(self):
        """Physical-group name holding the driven elements, or None.

        None means "use the default name", not "there is no driven group" —
        see bemkit.geometry.io.resolve_vibrating_group.
        """
        return self.groups_cfg.get("vibrating")

    def variants(self, prefixes=None):
        """Discover variant names from [variants] table or mesh/ directory."""
        from bemkit.sim.discovery import discover_variants
        return discover_variants(self.model_dir, prefixes)

    def mesh_map(self):
        """Return {variant: mesh_stem} from [variants] table, or {}.

        Generated variants (mesh overrides / source) map to their own name:
        their mesh is written as mesh/<variant>.msh.
        """
        tbl = self.raw.get("variants") or {}
        out = {}
        for n, v in tbl.items():
            m = v.get("mesh")
            out[n] = m if isinstance(m, str) else n
        return out

    def variant_source(self, variant):
        """Explicit geometry/ source stem for a variant, or None."""
        v = (self.raw.get("variants") or {}).get(variant, {})
        return v.get("source")

    def variant_mesh_params(self, variant):
        """Merged [mesh] + per-variant overrides for a generated variant.

        Returns None when the variant uses a trusted mesh/ file as-is
        (mesh key is a string, or absent without a source).
        """
        v = (self.raw.get("variants") or {}).get(variant, {})
        m = v.get("mesh")
        if isinstance(m, dict):
            return {**self.mesh_cfg, **m}
        if m is None and "source" in v:
            return dict(self.mesh_cfg)
        return None


def freq_grid(sr, nbins):
    """The simulated frequency grid, DC excluded — shape (nbins-1,).

    Every bin index in the toolkit refers to a position in this array. A
    partial-band run and the full run that later fills it in must resolve
    against the identical grid, or their results would merge at mismatched
    frequencies — so the formula lives in exactly one place.
    """
    return np.linspace(0, sr / 2, nbins)[1:]


def freq_bin(freq_hz, sr, nbins):
    """Return (0-based index, 1-based be.out index, actual_freq_hz) of nearest bin."""
    freqs = freq_grid(sr, nbins)
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    return idx, idx + 1, freqs[idx]


def resolve_bins(frequencies, *, freq=None, bins=None, preview=None):
    """Resolve a partial-band selection to sorted unique 1-based bin indices.

    A partial run computes a subset of the same grid a full run would, which is
    what lets its results merge into a later full run. Requested frequencies are
    therefore snapped onto that grid, never honoured exactly. Exactly one
    selector must be given.

    Args:
        frequencies: The model's frequency array in Hz — ``ModelConfig
            .freq_array()``. Taken as data rather than derived from sr/nbins,
            so a non-uniform ``[frequencies]`` band selects the same way.
        freq: Sequence of frequencies in Hz, each snapped to the nearest bin.
            Ties resolve to the lower bin. A frequency that is negative,
            non-finite, or above sr/2 raises rather than snapping silently.
        bins: Sequence of literal 1-based bin indices.
        preview: N bins spread logarithmically over the band. Log spacing
            because a linear pick from a linear grid says nothing about the
            bottom octaves; preview=1 resolves to the lowest bin. Bins that
            coincide after snapping collapse, so the result may be shorter
            than N.

    Returns:
        Sorted, unique, 1-based bin indices.
    """
    given = [name for name, val in
             (("freq", freq), ("bins", bins), ("preview", preview))
             if val is not None]
    if len(given) != 1:
        raise ValueError(
            f"resolve_bins needs exactly one of freq/bins/preview, got "
            f"{', '.join(given) if given else 'none'}")

    freqs = np.asarray(frequencies, dtype=float)
    nf = len(freqs)
    if nf == 0:
        raise ValueError(
            "no frequency bins to select from (a model needs at least one "
            "simulated frequency)")

    if bins is not None:
        wanted = np.asarray(list(bins), dtype=float)
        if wanted.size == 0:
            raise ValueError("bins selector is empty")
        # ~isfinite is part of the mask, not a separate check: floor(inf) == inf,
        # so inf would pass an integrality test and then cast to INT_MIN, and
        # the range error below would name -9223372036854775808.
        non_integral = wanted[~np.isfinite(wanted) | (wanted != np.floor(wanted))]
        if non_integral.size:
            raise ValueError(
                f"bin indices must be whole numbers: "
                f"{', '.join(f'{v:g}' for v in non_integral)}")
        wanted_bins = wanted.astype(int)
        outside = wanted_bins[(wanted_bins < 1) | (wanted_bins > nf)]
        if outside.size:
            raise ValueError(
                f"outside the band: {', '.join(str(b) for b in outside)} — "
                f"this model has {nf} bins, valid indices 1..{nf} "
                f"(bins are 1-based, like NumCalc's -istart)")
        return sorted(set(wanted_bins.tolist()))

    if preview is not None:
        n = int(preview)
        if n < 1:
            raise ValueError(f"preview needs at least 1 bin, got {n}")
        wanted_hz = np.geomspace(freqs[0], freqs[-1], n)
    else:
        wanted_hz = np.asarray([float(f) for f in freq], dtype=float)
        if wanted_hz.size == 0:
            raise ValueError("freq selector is empty")
        bad_mask = ~np.isfinite(wanted_hz) | (wanted_hz < 0)
        if bad_mask.any():
            bad = wanted_hz[bad_mask]
            raise ValueError(
                f"not valid as a frequency (must be finite and non-negative): "
                f"{', '.join(f'{v:g}' for v in bad)}")
        above = wanted_hz[wanted_hz > freqs[-1]]
        if above.size:
            raise ValueError(
                f"above the simulated band: "
                f"{', '.join(f'{v:g}' for v in above)} Hz — the band reaches "
                f"{freqs[-1]:g} Hz")

    return sorted({int(np.argmin(np.abs(freqs - f))) + 1 for f in wanted_hz})
