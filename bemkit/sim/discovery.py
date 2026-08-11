"""Discovery and loading helpers for BEM model directories."""

from pathlib import Path

from bemkit.config import ModelConfig
from bemkit.meshing.io import MESH_EXTENSIONS
from bemkit.numcalc.packing import has_results


def numcalc_dir(model_dir, variant):
    """Return the numcalc directory for a variant."""
    if variant is None:
        return Path(model_dir) / "numcalc"
    return Path(model_dir) / "numcalc" / variant


def _explicit_variant_names(model_dir):
    """Names from [variants] in config.toml, or None if absent/no config.toml."""
    if not (Path(model_dir) / "config.toml").exists():
        return None
    tbl = ModelConfig.from_dir(model_dir).get("variants")
    return list(tbl.keys()) if tbl else None


def discover_variants(model_dir, prefixes=None, *, need_be_out=False,
                      need_object_mesh=False):
    """Discover variants from mesh/ dir or [variants] table with optional filtering.

    Parameters
    ----------
    model_dir : Path
    prefixes : list of str or None
        Prefix match filters.
    need_be_out : bool
        If True, only include variants with results (be.out/ or be.npz).
    need_object_mesh : bool
        If True, only include variants with numcalc/{v}/ObjectMesh/Nodes.txt.

    Returns
    -------
    list of str
    """
    model_dir = Path(model_dir)
    explicit = _explicit_variant_names(model_dir)
    if explicit is not None:
        variants = explicit
    else:
        mesh_dir = model_dir / "mesh"
        if not mesh_dir.exists():
            return []
        mesh_files = sorted(f for f in mesh_dir.iterdir() if f.suffix in MESH_EXTENSIONS)
        variants = [f.stem for f in mesh_files]
    if prefixes:
        variants = [v for v in variants if any(v.startswith(p) for p in prefixes)]
    filtered = []
    for v in variants:
        if need_object_mesh:
            if not (model_dir / "numcalc" / v / "ObjectMesh" / "Nodes.txt").exists():
                continue
        if need_be_out:
            if not has_results(model_dir / "numcalc" / v):
                continue
        filtered.append(v)
    return filtered
