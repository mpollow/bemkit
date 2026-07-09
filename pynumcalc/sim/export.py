"""Per-variant NumCalc export: fieldpoints + mesh + NC.inp in one call."""

from dataclasses import dataclass, field
from pathlib import Path

from pynumcalc.meshing.io import MeshInfo, find_mesh_file, read_mesh_physical_groups
from pynumcalc.sim.fieldpoints import (
    combine_and_track, load_fieldpoint_sets, save_offsets, validate_points,
)
from pynumcalc.sim.writers import (
    write_eval_grid_elements, write_eval_grid_nodes, write_nc_inp,
    write_object_mesh_elements, write_object_mesh_nodes,
)


@dataclass(frozen=True)
class ExportInfo:
    """Per-variant output of the export stage."""

    variant: str
    nc_dir: Path
    mesh_info: MeshInfo
    n_nodes: int
    n_elements: int
    n_eval_points: int
    fieldpoint_sets: list[str] = field(default_factory=list)


def export_variant(config, variant, mesh_file=None, verbose=True):
    """Export fieldpoints, ObjectMesh, and NC.inp for one variant.

    Args:
        config: ModelConfig
        variant: variant name (numcalc/{variant}/ is created)
        mesh_file: mesh path override; default resolves via config.mesh_map()

    Returns:
        ExportInfo

    Raises:
        ValueError: no [fieldpoints] configured
        FileNotFoundError: mesh file missing
    """
    model_dir = config.model_dir
    nc_dir = model_dir / "numcalc" / variant
    if mesh_file is None:
        mesh_stem = config.mesh_map().get(variant, variant)
        mesh_file = find_mesh_file(model_dir / "mesh", mesh_stem)
    nc_dir.mkdir(parents=True, exist_ok=True)

    # -- fieldpoints --
    fp_config = config.fieldpoints_cfg
    if not fp_config:
        raise ValueError("No [fieldpoints] in vars.toml")
    sets = load_fieldpoint_sets(fp_config, model_dir)
    valid_masks = {}
    for name, points in sets.items():
        mask = validate_points(points, mesh_file, is_internal=config.is_internal,
                               edge_length_factor=0.1)
        if verbose:
            n = f"{mask.sum()}/{len(mask)}" if not mask.all() else f"all {len(points)}"
            print(f"  '{name}': {n} points valid")
        if not mask.all():
            valid_masks[name] = mask
    combined, offsets = combine_and_track(sets, valid_masks or None)
    if len(combined) == 0:
        raise ValueError(
            f"No valid evaluation points for variant '{variant}': all "
            f"{sum(len(p) for p in sets.values())} fieldpoint(s) were dropped by "
            f"validation (inside the mesh or too close to the surface). "
            f"Check the fieldpoint sets against the mesh.")
    write_eval_grid_nodes(combined, nc_dir)
    write_eval_grid_elements(len(combined), nc_dir)
    save_offsets(offsets, nc_dir / "eval_offsets.json")
    if verbose:
        print(f"EvalGrid: {len(combined)} nodes → {nc_dir}")

    # -- mesh --
    n_nodes = write_object_mesh_nodes(mesh_file, nc_dir)
    n_elements = write_object_mesh_elements(mesh_file, nc_dir)
    if verbose:
        print(f"ObjectMesh: {n_nodes} nodes, {n_elements} elements → {nc_dir}")

    # -- NC.inp --
    # Resolve BOUNDARY entries (including geometric near= anchors) against the
    # mesh, then write the concrete element ranges.
    from pynumcalc.sim.boundary import flatten, resolve_boundary
    groups = read_mesh_physical_groups(mesh_file)
    resolved = resolve_boundary(config, variant, mesh_file)
    write_nc_inp(config, variant, nc_dir,
                 vibrating_indices=groups.get("vibrating"),
                 boundary_ranges=flatten(resolved), verbose=verbose)

    # -- config snapshot (records the effective config + resolved BC) --
    from pynumcalc.sim.config_snapshot import write_snapshot
    write_snapshot(nc_dir, config, mesh_file=mesh_file, resolved_boundary=resolved)

    return ExportInfo(
        variant=variant,
        nc_dir=nc_dir,
        mesh_info=MeshInfo.from_file(mesh_file),
        n_nodes=n_nodes,
        n_elements=n_elements,
        n_eval_points=len(combined),
        fieldpoint_sets=list(sets.keys()),
    )
