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
    if missing:
        raise ValueError(f"{path}: [bem] is missing required key(s): "
                         f"{', '.join(missing)}")
    unknown = [k for k in bem if k not in _BEM_KEYS]
    if unknown:
        raise ValueError(f"{path}: unknown [bem] key(s): {', '.join(unknown)} "
                         f"(known: {', '.join(_BEM_KEYS)})")

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


@dataclass(frozen=True)
class ModelConfig:
    """Typed wrapper around a model directory's config.toml."""

    model_dir: Path
    modelname: str
    sr: int
    nbins: int
    air_density: float = 1.2
    speed_of_sound: float = 340.0
    bem_type: int = 4
    time_factor: float = -1.0
    is_internal: int = 0
    mesh_cfg: dict = field(default_factory=dict)
    geometry_cfg: dict = field(default_factory=dict)
    fieldpoints_cfg: dict = field(default_factory=dict)
    groups_cfg: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, model_dir):
        """Load config.toml; modelname defaults to the directory name."""
        model_dir = Path(model_dir).resolve()
        cfg_path = model_dir / "config.toml"
        with open(cfg_path, "rb") as f:
            raw = tomllib.load(f)
        _validate(raw, cfg_path)
        bem = raw["bem"]
        raw.setdefault("modelname", model_dir.name)
        return cls(
            model_dir=model_dir,
            modelname=raw["modelname"],
            sr=bem["sr"],
            nbins=bem["nbins"],
            air_density=bem.get("air_density", 1.2),
            speed_of_sound=bem.get("speed_of_sound", 340.0),
            bem_type=bem.get("bem_type", 4),
            time_factor=bem.get("time_factor", -1.0),
            is_internal=bem.get("is_internal", 0),
            mesh_cfg=raw.get("mesh", {}),
            geometry_cfg=raw.get("geometry", {}),
            fieldpoints_cfg=raw.get("fieldpoints", {}),
            groups_cfg=raw.get("groups", {}),
            raw=raw,
        )

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
        """Frequency array excluding DC, shape (nbins-1,)."""
        return freq_grid(self.sr, self.nbins)

    def vibrating_group(self):
        """Physical-group name holding the driven elements, or None.

        None means "use the default name", not "there is no driven group" —
        see bemkit.meshing.io.resolve_vibrating_group.
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


def resolve_bins(sr, nbins, *, freq=None, bins=None, preview=None):
    """Resolve a partial-band selection to sorted unique 1-based bin indices.

    A partial run computes a subset of the same linear grid a full run would,
    which is what lets its results merge into a later full run. Requested
    frequencies are therefore snapped onto that grid, never honoured exactly.
    Exactly one selector must be given.

    Args:
        sr: Sample rate in Hz.
        nbins: Bin count as configured in [bem] (n_frequencies = nbins - 1).
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

    freqs = freq_grid(sr, nbins)
    nf = len(freqs)
    if nf == 0:
        raise ValueError(
            f"nbins={nbins} leaves no frequency bins to select from (must "
            f"be at least 2: the DC bin plus one simulated bin)")

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
                f"{freqs[-1]:g} Hz for sr={sr}, nbins={nbins}")

    return sorted({int(np.argmin(np.abs(freqs - f))) + 1 for f in wanted_hz})
