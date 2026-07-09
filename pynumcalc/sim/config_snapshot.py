"""Per-bundle config snapshot.

At export time the effective configuration that produced a ``numcalc/{variant}/``
bundle is written to ``config.json`` inside it. Result loaders prefer this
snapshot over the live ``vars.toml`` so that editing ``sr``/``nbins`` (or the
air properties) after a run cannot silently reinterpret already-computed results
at the wrong frequencies. A mismatch between the snapshot and the live config is
surfaced as a warning rather than silently followed.
"""

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_NAME = "config.json"

# Fields whose value defines how results are interpreted; a drift in any of
# these between snapshot and live config is worth warning about.
_INTERPRETIVE_KEYS = ("sr", "nbins", "air_density", "speed_of_sound")

# Every scalar recorded in the snapshot.
_SNAPSHOT_KEYS = _INTERPRETIVE_KEYS + (
    "modelname", "bem_type", "time_factor", "is_internal")


def _pynumcalc_version():
    try:
        from importlib.metadata import version
        return version("pynumcalc")
    except Exception:  # dev checkout without installed metadata
        return "unknown"


def _mesh_sha256(mesh_file):
    try:
        h = hashlib.sha256()
        with open(mesh_file, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def write_snapshot(nc_dir, config, *, mesh_file=None, resolved_boundary=None):
    """Write ``nc_dir/config.json`` capturing the effective config.

    Parameters
    ----------
    nc_dir : Path or str
    config : ModelConfig or dict-like
        Anything supporting ``config.get(key, default)``.
    mesh_file : Path or str or None
        Mesh the bundle was exported from (name + content hash recorded).
    resolved_boundary : dict or None
        Resolved ``{key: [[start, end, velocity], ...]}`` element ranges (from
        geometric BC resolution), recorded for provenance.
    """
    nc_dir = Path(nc_dir)
    snap = {k: config.get(k) for k in _SNAPSHOT_KEYS}
    snap["pynumcalc_version"] = _pynumcalc_version()
    snap["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if mesh_file is not None:
        mesh_file = Path(mesh_file)
        snap["mesh_file"] = mesh_file.name
        snap["mesh_sha256"] = _mesh_sha256(mesh_file)
    if resolved_boundary is not None:
        snap["resolved_boundary"] = {
            k: [list(r) for r in ranges] for k, ranges in resolved_boundary.items()}
    with open(nc_dir / SNAPSHOT_NAME, "w") as f:
        json.dump(snap, f, indent=2)
    return snap


def read_snapshot(nc_dir):
    """Return the bundle's ``config.json`` as a dict, or None if absent."""
    path = Path(nc_dir) / SNAPSHOT_NAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def warn_on_drift(snapshot, config, nc_dir=None):
    """Warn if the live config disagrees with the snapshot on interpretive keys."""
    drift = []
    for key in _INTERPRETIVE_KEYS:
        snap_val = snapshot.get(key)
        live_val = config.get(key)
        if snap_val is not None and live_val is not None and snap_val != live_val:
            drift.append(f"{key}: exported {snap_val!r} vs vars.toml {live_val!r}")
    if drift:
        where = f" for {nc_dir}" if nc_dir is not None else ""
        warnings.warn(
            f"Bundle config.json differs from current vars.toml{where}: "
            f"{'; '.join(drift)}. Results reflect what was exported (config.json); "
            f"re-export to apply the new values.",
            stacklevel=2,
        )
